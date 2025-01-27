import os
import sys
# 获取当前脚本的绝对路径
current_script_path = os.path.abspath(__file__)

# 将项目根目录添加到sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_script_path))))
root_dir = os.path.dirname(root_dir)
sys.path.append(root_dir)
from src.configs.configs import (MYSQL_HOST_LOCAL, MYSQL_PORT_LOCAL, MYSQL_USER_LOCAL,
                                                   MYSQL_PASSWORD_LOCAL,
                                                   MYSQL_DATABASE_LOCAL, KB_SUFFIX, MILVUS_HOST_LOCAL)
from src.utils.log_handler import debug_logger, insert_logger
import mysql.connector
from mysql.connector import pooling
import json
from typing import List, Optional, Dict
import uuid
from datetime import datetime, timedelta
from collections import defaultdict
from mysql.connector.errors import Error as MySQLError
class MysqlClient:
    def __init__(self, pool_size=8):
        host = MYSQL_HOST_LOCAL
        port = MYSQL_PORT_LOCAL
        user = MYSQL_USER_LOCAL
        password = MYSQL_PASSWORD_LOCAL
        database = MYSQL_DATABASE_LOCAL

        self.check_database_(host, port, user, password, database)
        dbconfig = {
            "host": host,
            "user": user,
            "port": port,
            "password": password,
            "database": database,
        }
        self.cnxpool = pooling.MySQLConnectionPool(pool_size=pool_size, pool_reset_session=True, **dbconfig)
        self.free_cnx = pool_size
        self.used_cnx = 0
        self.create_tables_()
        debug_logger.info("[SUCCESS] 数据库{}连接成功".format(database))    

    def check_database_(self, host, port, user, password, database_name):
        # 连接 MySQL 服务器
        cnx = mysql.connector.connect(
            host=host,
            port=port,
            user=user,
            password=password
        )

        # 检查数据库是否存在
        cursor = cnx.cursor(buffered=True)
        cursor.execute('SHOW DATABASES')
        databases = [database[0] for database in cursor]

        if database_name not in databases:
            # 如果数据库不存在，则新建数据库
            cursor.execute('CREATE DATABASE IF NOT EXISTS {}'.format(database_name))
            debug_logger.info("数据库{}新建成功或已存在".format(database_name))
        debug_logger.info("[SUCCESS] 数据库{}检查通过".format(database_name))
        # 关闭游标
        cursor.close()
        # 连接到数据库
        cnx.database = database_name
        # 关闭数据库连接
        cnx.close()

    def execute_query_(self, query, params, commit=False, fetch=False, check=False, user_dict=False):
        try:
            conn = self.cnxpool.get_connection()
            self.used_cnx += 1
            self.free_cnx -= 1
            if self.free_cnx < 4:
                debug_logger.info("获取连接成功，当前连接池状态：空闲连接数 {}，已使用连接数 {}".format(
                    self.free_cnx, self.used_cnx))
        except MySQLError as err:
            debug_logger.error("从连接池获取连接失败：{}".format(err))
            return None

        result = None
        cursor = None
        try:
            if user_dict:
                cursor = conn.cursor(dictionary=True)
            else:
                cursor = conn.cursor(buffered=True)
            cursor.execute(query, params)

            if commit:
                conn.commit()

            if fetch:
                result = cursor.fetchall()
            elif check:
                result = cursor.rowcount
        except MySQLError as err:
            if err.errno == 1061:
                debug_logger.info(f"Index already exists (this is okay): {query}")
            else:
                debug_logger.error("执行数据库操作失败：{}，SQL：{}".format(err, query))
            if commit:
                conn.rollback()
        finally:
            if cursor is not None:
                cursor.close()
            conn.close()
            self.used_cnx -= 1
            self.free_cnx += 1
            if self.free_cnx <= 4:
                debug_logger.info("连接关闭，返回连接池。当前连接池状态：空闲连接数 {}，已使用连接数 {}".format(
                    self.free_cnx, self.used_cnx))

        return result
    # 数据库建表语句
    def create_tables_(self):
        query = """
            CREATE TABLE IF NOT EXISTS User (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_id VARCHAR(255) UNIQUE,
                user_name VARCHAR(255),
                creation_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """

        self.execute_query_(query, (), commit=True)
        query = """
            CREATE TABLE IF NOT EXISTS KnowledgeBase (
                id INT AUTO_INCREMENT PRIMARY KEY,
                kb_id VARCHAR(255) UNIQUE,
                user_id VARCHAR(255),
                kb_name VARCHAR(255),
                deleted BOOL DEFAULT 0,
                latest_qa_time TIMESTAMP,
                latest_insert_time TIMESTAMP
            );

        """
        self.execute_query_(query, (), commit=True)
        query = """
            CREATE TABLE IF NOT EXISTS File (
                id INT AUTO_INCREMENT PRIMARY KEY,
                file_id VARCHAR(255) UNIQUE,
                user_id VARCHAR(255) DEFAULT 'unknown',
                kb_id VARCHAR(255),
                file_name VARCHAR(255),
                status VARCHAR(255),
                msg VARCHAR(255) DEFAULT 'success',
                transfer_status VARCHAR(255),
                deleted BOOL DEFAULT 0,
                file_size INT DEFAULT -1,
                content_length INT DEFAULT -1,
                chunks_number INT DEFAULT -1,
                file_location VARCHAR(255) DEFAULT 'unknown',
                file_url VARCHAR(2048) DEFAULT '',
                upload_infos TEXT,
                chunk_size INT DEFAULT -1,
                timestamp VARCHAR(255) DEFAULT '197001010000'
            );

        """
        self.execute_query_(query, (), commit=True)

        # create_index_query = "CREATE INDEX IF NOT EXISTS index_kb_id_deleted ON File (kb_id, deleted);"
        # self.execute_query_(create_index_query, (), commit=True)
        # create_index_query = "CREATE INDEX idx_user_id_status ON File (user_id, status);"
        # self.execute_query_(create_index_query, (), commit=True)

        query = """
            CREATE TABLE IF NOT EXISTS Faqs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                faq_id  VARCHAR(255) UNIQUE,
                user_id VARCHAR(255) NOT NULL,
                kb_id VARCHAR(255) NOT NULL,
                question VARCHAR(512) NOT NULL,
                answer VARCHAR(2048) NOT NULL,
                nos_keys VARCHAR(768)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        self.execute_query_(query, (), commit=True)

        query = """
            CREATE TABLE IF NOT EXISTS Documents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                doc_id VARCHAR(255) UNIQUE,
                json_data LONGTEXT
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """

        self.execute_query_(query, (), commit=True)
        # 创建一个QaLogs表，用于记录用户的操作日志
        """
        chat_data = {'user_id': user_id, 'kb_ids': kb_ids, 'query': question, "model": model, "product_source": request_source,
                     'time_record': time_record, 'history': history,
                     'condense_question': resp['condense_question'],
                     'prompt': resp['prompt'], 'result': next_history[-1][1],
                     'retrieval_documents': retrieval_documents, 'source_documents': source_documents}
        """
        # 其中kb_ids是一个List[str], time_record是Dict，history是List[List[str]], retrieval_documents是List[Dict], source_documents是List[Dict]，其他项都是str
        query = """
            CREATE TABLE IF NOT EXISTS QaLogs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                qa_id VARCHAR(255) UNIQUE,
                user_id VARCHAR(255) NOT NULL,
                kb_ids VARCHAR(2048) NOT NULL,
                query VARCHAR(512) NOT NULL,
                model VARCHAR(64) NOT NULL,
                product_source VARCHAR(64) NOT NULL,
                time_record VARCHAR(512) NOT NULL,
                history MEDIUMTEXT NOT NULL,
                condense_question VARCHAR(1024) NOT NULL,
                prompt MEDIUMTEXT NOT NULL,
                result TEXT NOT NULL,
                retrieval_documents MEDIUMTEXT NOT NULL,
                source_documents MEDIUMTEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        self.execute_query_(query, (), commit=True)

        # create_index_query = "CREATE INDEX IF NOT EXISTS index_bot_id ON QaLogs (bot_id);"
        # self.execute_query_(create_index_query, (), commit=True)
        # create_index_query = "CREATE INDEX IF NOT EXISTS index_query ON QaLogs (query);"
        # self.execute_query_(create_index_query, (), commit=True)
        # create_index_query = "CREATE INDEX IF NOT EXISTS index_timestamp ON QaLogs (timestamp);"
        # self.execute_query_(create_index_query, (), commit=True)
        # 存储图片
        query = """
            CREATE TABLE IF NOT EXISTS FileImages (
                id INT AUTO_INCREMENT PRIMARY KEY,
                image_id VARCHAR(255) UNIQUE,
                file_id VARCHAR(255) NOT NULL,
                user_id VARCHAR(255) NOT NULL,
                kb_id VARCHAR(255) NOT NULL,
                nos_key VARCHAR(255) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
        """
        self.execute_query_(query, (), commit=True)


        # 修改索引创建方式
        index_queries = [
            "CREATE INDEX index_kb_id_deleted ON File (kb_id, deleted)",
            "CREATE INDEX idx_user_id_status ON File (user_id, status)",
            "CREATE INDEX index_query ON QaLogs (query)",
            "CREATE INDEX index_timestamp ON QaLogs (timestamp)",
        ]

        for query in index_queries:
            try:
                self.execute_query_(query, (), commit=True)
                debug_logger.info(f"Index created successfully: {query}")
            except mysql.connector.Error as err:
                if err.errno == 1061:  # 重复键错误
                    debug_logger.info(f"Index already exists (this is okay): {query}")
                elif err.errno == 1060:  # 已存在的列无需创建
                    debug_logger.info(f"Column already exists (this is okay): {query}")
                elif err.errno == 1091:  # 已经删除的列无需删除
                    debug_logger.info(f"Column already deleted (this is okay): {query}")
                else:
                    debug_logger.error(f"Error creating index: {err}")

        debug_logger.info("All tables and indexes checked/created successfully.")
    # 检查知识库是否存在
    def check_kb_exist(self, user_id, kb_ids):
        if not kb_ids:
            return []
        kb_ids_str = ','.join("'{}'".format(str(x)) for x in kb_ids)
        query = "SELECT kb_id FROM KnowledgeBase WHERE kb_id IN ({}) AND deleted = 0 AND user_id = %s".format(
            kb_ids_str)
        result = self.execute_query_(query, (user_id,), fetch=True)
        debug_logger.info("check_kb_exist {}".format(result))
        valid_kb_ids = [kb_info[0] for kb_info in result]
        unvalid_kb_ids = list(set(kb_ids) - set(valid_kb_ids))
        return unvalid_kb_ids
    # 检查用户是否存在
    def check_user_exist_(self, user_id):
        query = "SELECT user_id FROM User WHERE user_id = %s"
        result = self.execute_query_(query, (user_id,), fetch=True)
        debug_logger.info("check_user_exist {}".format(result))
        return result is not None and len(result) > 0

    # 对外接口不需要增加用户，新建知识库的时候增加用户就可以了
    def add_user_(self, user_id, user_name):
        query = "INSERT IGNORE INTO User (user_id, user_name) VALUES (%s, %s)"
        self.execute_query_(query, (user_id, user_name), commit=True)
        debug_logger.info(f"Add user: {user_id} {user_name}")
    # 创建新的知识库
    def new_milvus_base(self, kb_id, user_id, kb_name, user_name=None):
        if not self.check_user_exist_(user_id):
            self.add_user_(user_id, user_name)
        query = "INSERT INTO KnowledgeBase (kb_id, user_id, kb_name) VALUES (%s, %s, %s)"
        self.execute_query_(query, (kb_id, user_id, kb_name), commit=True)
        return kb_id, "success"
    # 获取知识库中的文件
    def get_files(self, kb_id, file_id=None):
        limit = 100
        offset = 0
        all_files = []

        base_query = """
            SELECT file_id, file_name, status, file_size, content_length, timestamp,
                   file_location, file_url, chunk_size, msg
            FROM File
            WHERE kb_id = %s AND deleted = 0
        """

        params = [kb_id]
        if file_id is not None:
            base_query += " AND file_id=%s"
            params.append(file_id)
            files = self.execute_query_(base_query, params, fetch=True)
            return files
        # 更好的利用数据库连接池中的多个连接
        while True:
            query = base_query + "LIMIT %s AND OFFSET %s"
            current_params = params + [limit, offset]
            files = self.execute_query_(query, current_params, fetch=True)
            if not files:
                break
            all_files.extend(files)
            offset += limit
        
        return all_files
    # 查看文件是否存在
    def check_file_exist_by_name(self, user_id, kb_id, file_names):
        results = []
        batch_size = 100  # 根据实际情况调整批次大小

        # 分批处理file_names
        for i in range(0, len(file_names), batch_size):
            batch_file_names = file_names[i:i + batch_size]

            # 创建参数化的查询，用%s作为占位符
            placeholders = ','.join(['%s'] * len(batch_file_names))
            query = """
                SELECT file_id, file_name, file_size, status FROM File
                WHERE deleted = 0
                AND file_name IN ({})
                AND kb_id = %s
                AND kb_id IN (SELECT kb_id FROM KnowledgeBase WHERE user_id = %s)
            """.format(placeholders)

            # 使用参数化查询，将文件名作为参数传递
            query_params = batch_file_names + [kb_id, user_id]
            batch_result = self.execute_query_(query, query_params, fetch=True)
            debug_logger.info("check_file_exist_by_name batch {}: {}".format(i // batch_size, batch_result))
            results.extend(batch_result)

        return results
    
        # [知识库] 获取指定kb_ids的知识库
    # 获取知识库名字
    def get_knowledge_base_name(self, kb_ids):
        kb_ids_str = ','.join("'{}'".format(str(x)) for x in kb_ids)
        query = "SELECT user_id, kb_id, kb_name FROM KnowledgeBase WHERE kb_id IN ({}) AND deleted = 0".format(
            kb_ids_str)
        return self.execute_query_(query, (), fetch=True)
    
    # [文件] 向指定知识库下面增加文件
    def add_file(self, file_id, user_id, kb_id, file_name, file_size, file_location, chunk_size, timestamp, file_url='',
                 status="gray"):
        query = ("INSERT INTO File (file_id, user_id, kb_id, file_name, status, file_size, file_location, chunk_size, "
                 "timestamp, file_url) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)")
        self.execute_query_(query,
                            (file_id, user_id, kb_id, file_name, status, file_size, file_location, chunk_size, timestamp, file_url),
                            commit=True)
        
    # [文件] 添加 chunks number 字段
    def modify_file_chunks_number(self, file_id, user_id, kb_id, chunks_number):
        query = ("UPDATE File SET chunks_number = %s WHERE file_id = %s AND user_id = %s AND kb_id = %s")
        self.execute_query_(query, (chunks_number, file_id, user_id, kb_id), commit=True)