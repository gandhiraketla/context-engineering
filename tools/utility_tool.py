import json
from datetime import datetime, timedelta
import random

class UtilityTools:
    """DevOps utility tools for S3 management, system operations, and security tasks"""
    
    def list_s3_buckets(self) -> list:
        """
        List all S3 buckets in the current AWS account.
        
        Returns:
            list: List of dictionaries containing S3 bucket information
        """
        return [
            {"name": "prod-app-backups", "creation_date": "2023-06-15", "region": "us-east-1", "size": "45.2 GB"},
            {"name": "static-website-assets", "creation_date": "2023-08-20", "region": "us-east-1", "size": "12.8 GB"},
            {"name": "log-storage-bucket", "creation_date": "2023-07-10", "region": "us-west-2", "size": "128.5 GB"},
            {"name": "dev-deployment-artifacts", "creation_date": "2023-09-05", "region": "us-east-1", "size": "8.3 GB"}
        ]
    
    def check_s3_file(self, bucket: str, key: str) -> dict:
        """
        Check if a file exists in an S3 bucket and get its metadata.
        
        Args:
            bucket (str): The S3 bucket name
            key (str): The file key/path in the bucket
            
        Returns:
            dict: File metadata including size, modification date, and storage class
        """
        return {
            "bucket": bucket,
            "key": key,
            "exists": True,
            "size": f"{random.randint(1, 1000)} MB",
            "last_modified": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d %H:%M:%S"),
            "storage_class": random.choice(["STANDARD", "STANDARD_IA", "GLACIER"]),
            "etag": f'"{random.randint(100000000, 999999999):x}"',
            "content_type": "application/octet-stream",
            "encryption": "AES256"
        }
    
    def delete_s3_file(self, bucket: str, key: str) -> str:
        """
        Delete a file from an S3 bucket.
        
        Args:
            bucket (str): The S3 bucket name
            key (str): The file key/path to delete
            
        Returns:
            str: Confirmation message of deletion operation
        """
        return f"File {key} successfully deleted from bucket {bucket}. Deletion timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}. This action cannot be undone."
    
    def check_s3_bucket_policy(self, bucket: str) -> dict:
        """
        Check the bucket policy configuration for an S3 bucket.
        
        Args:
            bucket (str): The S3 bucket name
            
        Returns:
            dict: Bucket policy configuration and security settings
        """
        return {
            "bucket": bucket,
            "public_access_blocked": True,
            "bucket_policy_exists": True,
            "versioning": "Enabled",
            "mfa_delete": "Disabled",
            "logging": {
                "enabled": True,
                "target_bucket": f"{bucket}-access-logs",
                "target_prefix": "access-logs/"
            },
            "encryption": {
                "enabled": True,
                "type": "SSE-S3",
                "kms_key": "aws/s3"
            },
            "lifecycle_rules": 2
        }
    
    def get_s3_bucket_metrics(self, bucket: str) -> dict:
        """
        Get storage metrics and statistics for an S3 bucket.
        
        Args:
            bucket (str): The S3 bucket name
            
        Returns:
            dict: Bucket metrics including size, object count, and costs
        """
        return {
            "bucket": bucket,
            "total_size": f"{random.uniform(10, 500):.1f} GB",
            "object_count": random.randint(1000, 50000),
            "storage_classes": {
                "STANDARD": f"{random.randint(60, 80)}%",
                "STANDARD_IA": f"{random.randint(10, 25)}%",
                "GLACIER": f"{random.randint(5, 15)}%"
            },
            "monthly_cost": f"${random.uniform(50, 500):.2f}",
            "requests_last_month": random.randint(10000, 100000),
            "data_transfer": f"{random.uniform(5, 50):.1f} GB"
        }
    
    def replicate_s3_bucket(self, source: str, target: str) -> str:
        """
        Set up replication between S3 buckets.
        
        Args:
            source (str): Source bucket name
            target (str): Target bucket name for replication
            
        Returns:
            str: Status message of replication setup
        """
        return f"Cross-region replication configured from {source} to {target}. Replication rule created with prefix 'data/'. Initial sync estimated: 2-6 hours depending on data size."
    
    def restart_service(self, service_name: str) -> str:
        """
        Restart a system service.
        
        Args:
            service_name (str): Name of the service to restart
            
        Returns:
            str: Service restart status and timing information
        """
        return f"Service {service_name} restart initiated. Status: stopping -> stopped -> starting -> running. Estimated restart time: 30-45 seconds. Service will be available shortly."
    
    def list_running_processes(self, server_id: str) -> list:
        """
        List running processes on a server.
        
        Args:
            server_id (str): The server identifier
            
        Returns:
            list: List of running processes with resource usage
        """
        return [
            {"pid": 1234, "name": "nginx", "cpu": f"{random.randint(1, 15)}%", "memory": f"{random.randint(50, 200)} MB", "user": "www-data"},
            {"pid": 5678, "name": "postgres", "cpu": f"{random.randint(5, 25)}%", "memory": f"{random.randint(200, 800)} MB", "user": "postgres"},
            {"pid": 9012, "name": "python3", "cpu": f"{random.randint(10, 40)}%", "memory": f"{random.randint(100, 500)} MB", "user": "app"},
            {"pid": 3456, "name": "redis-server", "cpu": f"{random.randint(2, 12)}%", "memory": f"{random.randint(80, 300)} MB", "user": "redis"},
            {"pid": 7890, "name": "docker", "cpu": f"{random.randint(5, 20)}%", "memory": f"{random.randint(150, 600)} MB", "user": "root"}
        ]
    
    def check_system_uptime(self, server_id: str) -> str:
        """
        Check system uptime for a server.
        
        Args:
            server_id (str): The server identifier
            
        Returns:
            str: System uptime information with load averages
        """
        uptime_days = random.randint(1, 100)
        uptime_hours = random.randint(0, 23)
        uptime_minutes = random.randint(0, 59)
        load_1 = random.uniform(0.5, 3.0)
        load_5 = random.uniform(0.4, 2.8)
        load_15 = random.uniform(0.3, 2.5)
        
        return f"""System uptime for {server_id}: {uptime_days} days, {uptime_hours} hours, {uptime_minutes} minutes
Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Load averages: {load_1:.2f}, {load_5:.2f}, {load_15:.2f}
Users logged in: {random.randint(1, 5)}
System load: {'Normal' if load_1 < 2.0 else 'High'}"""
    
    def rotate_access_keys(self, user: str) -> dict:
        """
        Rotate access keys for a user account.
        
        Args:
            user (str): Username for access key rotation
            
        Returns:
            dict: New access key information and rotation status
        """
        new_access_key = f"AKIA{random.randint(100000000000000, 999999999999999)}"
        return {
            "user": user,
            "operation": "access_key_rotation",
            "old_key_status": "deactivated",
            "new_access_key_id": new_access_key,
            "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "active",
            "secret_key_last_4": f"****{random.randint(1000, 9999)}",
            "rotation_schedule": "90 days",
            "next_rotation": (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")
        }