import json
from datetime import datetime, timedelta
import random

class DatabaseTools:
    """DevOps tools for managing RDS instances and Lambda functions"""
    
    def get_rds_health(self, db_instance: str) -> dict:
        """
        Get the health status of an RDS database instance.
        
        Args:
            db_instance (str): The RDS instance identifier
            
        Returns:
            dict: Database health information including connections, performance metrics
        """
        return {
            "db_instance": db_instance,
            "status": "available",
            "engine": "PostgreSQL 13.7",
            "cpu_utilization": f"{random.randint(15, 75)}%",
            "database_connections": random.randint(12, 95),
            "max_connections": 100,
            "read_iops": random.randint(200, 1000),
            "write_iops": random.randint(100, 500),
            "free_storage": f"{random.randint(45, 85)} GB",
            "backup_retention": "7 days"
        }
    
    def get_rds_logs(self, db_instance: str, lines: int = 20) -> str:
        """
        Retrieve database logs from an RDS instance.
        
        Args:
            db_instance (str): The RDS instance identifier
            lines (int): Number of log lines to retrieve (default: 20)
            
        Returns:
            str: Multi-line string containing database logs
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        return f"""[{timestamp}] LOG: database system is ready to accept connections
[{timestamp}] LOG: autovacuum launcher started
[{timestamp}] LOG: checkpoint starting: time
[{timestamp}] WARNING: long-running query detected (15.2s): SELECT * FROM orders
[{timestamp}] LOG: checkpoint complete: wrote 234 buffers (1.4%); sync=0.012s
[{timestamp}] LOG: automatic analyze of table "public.users" completed"""
    
    def restart_rds_instance(self, db_instance: str) -> str:
        """
        Restart an RDS database instance.
        
        Args:
            db_instance (str): The RDS instance identifier to restart
            
        Returns:
            str: Confirmation message of restart operation
        """
        return f"RDS instance {db_instance} restart initiated. Expected downtime: 5-10 minutes. Status: available -> rebooting -> available"
    
    def backup_rds_instance(self, db_instance: str) -> str:
        """
        Create a manual snapshot backup of an RDS instance.
        
        Args:
            db_instance (str): The RDS instance identifier to backup
            
        Returns:
            str: Backup operation status with snapshot details
        """
        snapshot_id = f"{db_instance}-snapshot-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        return f"Manual snapshot {snapshot_id} created for {db_instance}. Size: {random.randint(50, 200)} GB. Estimated completion: 15-30 minutes."
    
    def list_rds_instances(self) -> list:
        """
        List all RDS database instances in the current region.
        
        Returns:
            list: List of dictionaries containing RDS instance information
        """
        return [
            {"db_instance": "prod-postgres-primary", "engine": "PostgreSQL", "status": "available", "size": "db.r5.xlarge"},
            {"db_instance": "prod-postgres-readonly", "engine": "PostgreSQL", "status": "available", "size": "db.r5.large"},
            {"db_instance": "staging-mysql", "engine": "MySQL", "status": "available", "size": "db.t3.medium"},
            {"db_instance": "dev-postgres", "engine": "PostgreSQL", "status": "stopped", "size": "db.t3.small"}
        ]
    
    def get_lambda_status(self, function_name: str) -> dict:
        """
        Get the status and configuration of a Lambda function.
        
        Args:
            function_name (str): Name of the Lambda function
            
        Returns:
            dict: Lambda function status and configuration details
        """
        return {
            "function_name": function_name,
            "state": "Active",
            "last_modified": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "runtime": "python3.9",
            "memory": f"{random.choice([128, 256, 512, 1024])} MB",
            "timeout": f"{random.randint(30, 900)} seconds",
            "invocations_24h": random.randint(100, 5000),
            "error_rate": f"{random.uniform(0.1, 2.5):.1f}%",
            "avg_duration": f"{random.randint(50, 2000)} ms"
        }
    
    def invoke_lambda_function(self, function_name: str, payload: dict) -> dict:
        """
        Invoke a Lambda function with the provided payload.
        
        Args:
            function_name (str): Name of the Lambda function
            payload (dict): Input payload for the function
            
        Returns:
            dict: Function execution result and metadata
        """
        execution_time = random.randint(100, 3000)
        return {
            "function_name": function_name,
            "status_code": 200,
            "execution_time": f"{execution_time} ms",
            "billed_duration": f"{execution_time + 50} ms",
            "memory_used": f"{random.randint(64, 512)} MB",
            "response": {"message": "Function executed successfully", "processed_items": random.randint(1, 100)},
            "log_group": f"/aws/lambda/{function_name}"
        }
    
    def list_lambda_functions(self) -> list:
        """
        List all Lambda functions in the current region.
        
        Returns:
            list: List of dictionaries containing Lambda function information
        """
        return [
            {"function_name": "data-processor", "runtime": "python3.9", "memory": "512 MB", "last_modified": "2024-01-15"},
            {"function_name": "email-sender", "runtime": "nodejs18.x", "memory": "256 MB", "last_modified": "2024-01-20"},
            {"function_name": "image-resizer", "runtime": "python3.9", "memory": "1024 MB", "last_modified": "2024-01-10"},
            {"function_name": "webhook-handler", "runtime": "python3.9", "memory": "128 MB", "last_modified": "2024-01-22"}
        ]
    
    def update_lambda_code(self, function_name: str, version: str) -> str:
        """
        Update the code of a Lambda function to a new version.
        
        Args:
            function_name (str): Name of the Lambda function
            version (str): New version identifier
            
        Returns:
            str: Status message of code update operation
        """
        return f"Lambda function {function_name} code updated to version {version}. New version ARN: arn:aws:lambda:us-east-1:123456789:function:{function_name}:{version}"
    
    def get_lambda_logs(self, function_name: str, lines: int = 20) -> str:
        """
        Retrieve execution logs from a Lambda function.
        
        Args:
            function_name (str): Name of the Lambda function
            lines (int): Number of log lines to retrieve (default: 20)
            
        Returns:
            str: Multi-line string containing Lambda function logs
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        request_id = f"{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}"
        return f"""[{timestamp}] START RequestId: {request_id} Version: $LATEST
[{timestamp}] INFO: Processing event with 5 records
[{timestamp}] INFO: Connected to database successfully
[{timestamp}] INFO: Processing record 1 of 5
[{timestamp}] INFO: Data validation completed
[{timestamp}] END RequestId: {request_id}
[{timestamp}] REPORT Duration: 1247.83 ms Billed Duration: 1300 ms Memory: 512 MB Max Memory Used: 128 MB"""