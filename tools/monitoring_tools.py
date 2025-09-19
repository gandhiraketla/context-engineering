import json
from datetime import datetime, timedelta
import random

class MonitoringTools:
    """DevOps tools for monitoring, alerting, and CI/CD pipeline management"""
    
    def get_cloudwatch_metrics(self, resource_id: str, metric: str) -> dict:
        """
        Retrieve CloudWatch metrics for a specific resource.
        
        Args:
            resource_id (str): The resource identifier
            metric (str): The metric name (e.g., 'CPUUtilization', 'NetworkIn')
            
        Returns:
            dict: Metric data with timestamps and values
        """
        base_time = datetime.now()
        datapoints = []
        for i in range(6):
            timestamp = base_time - timedelta(minutes=i*5)
            value = random.uniform(20, 80) if metric == 'CPUUtilization' else random.randint(1000, 10000)
            datapoints.append({
                "timestamp": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "value": round(value, 2),
                "unit": "Percent" if "Utilization" in metric else "Bytes"
            })
        
        return {
            "resource_id": resource_id,
            "metric_name": metric,
            "datapoints": datapoints,
            "period": 300,
            "statistic": "Average"
        }
    
    def get_cloudwatch_alarms(self) -> list:
        """
        List all CloudWatch alarms and their current states.
        
        Returns:
            list: List of dictionaries containing alarm information
        """
        return [
            {"alarm_name": "HighCPUUtilization-WebServer", "state": "ALARM", "reason": "Threshold Crossed", "updated": "2024-01-23T10:30:00Z"},
            {"alarm_name": "DatabaseConnections-High", "state": "OK", "reason": "Sufficient Data", "updated": "2024-01-23T09:45:00Z"},
            {"alarm_name": "DiskSpace-Low-Storage", "state": "INSUFFICIENT_DATA", "reason": "Insufficient Data", "updated": "2024-01-23T08:15:00Z"},
            {"alarm_name": "LoadBalancer-ResponseTime", "state": "OK", "reason": "Sufficient Data", "updated": "2024-01-23T11:00:00Z"}
        ]
    
    def acknowledge_alarm(self, alarm_id: str) -> str:
        """
        Acknowledge a CloudWatch alarm to suppress notifications.
        
        Args:
            alarm_id (str): The alarm identifier
            
        Returns:
            str: Confirmation message of alarm acknowledgment
        """
        return f"Alarm {alarm_id} acknowledged by operator. Notifications suppressed for 4 hours. Next evaluation: {(datetime.now() + timedelta(hours=4)).strftime('%Y-%m-%d %H:%M:%S')}"
    
    def check_disk_usage(self, server_id: str) -> dict:
        """
        Check disk usage statistics for a server.
        
        Args:
            server_id (str): The server identifier
            
        Returns:
            dict: Disk usage information for all mounted filesystems
        """
        return {
            "server_id": server_id,
            "filesystems": [
                {"mount": "/", "size": "20G", "used": f"{random.randint(8, 18)}G", "available": f"{random.randint(2, 12)}G", "use_percent": f"{random.randint(40, 90)}%"},
                {"mount": "/var/log", "size": "10G", "used": f"{random.randint(2, 8)}G", "available": f"{random.randint(2, 8)}G", "use_percent": f"{random.randint(20, 80)}%"},
                {"mount": "/tmp", "size": "5G", "used": f"{random.randint(1, 4)}G", "available": f"{random.randint(1, 4)}G", "use_percent": f"{random.randint(10, 80)}%"}
            ],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def get_audit_logs(self, resource_id: str, lines: int = 20) -> str:
        """
        Retrieve audit logs for a specific resource.
        
        Args:
            resource_id (str): The resource identifier
            lines (int): Number of log lines to retrieve (default: 20)
            
        Returns:
            str: Multi-line string containing audit log entries
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""[{timestamp}] AUDIT: User admin@company.com accessed resource {resource_id}
[{timestamp}] AUDIT: Configuration change applied to {resource_id} by deploy-service
[{timestamp}] AUDIT: Security scan completed for {resource_id} - No issues found
[{timestamp}] AUDIT: Backup operation initiated for {resource_id}
[{timestamp}] AUDIT: Resource {resource_id} scaled from 2 to 4 instances
[{timestamp}] AUDIT: SSL certificate renewed for {resource_id}"""
    
    def trigger_ci_cd_pipeline(self, pipeline_id: str) -> str:
        """
        Trigger a CI/CD pipeline execution.
        
        Args:
            pipeline_id (str): The pipeline identifier
            
        Returns:
            str: Pipeline execution status and build number
        """
        build_number = random.randint(100, 999)
        return f"Pipeline {pipeline_id} triggered successfully. Build #{build_number} started. Estimated completion: 8-12 minutes. View progress at /builds/{build_number}"
    
    def rollback_deployment(self, service_name: str) -> str:
        """
        Rollback a service deployment to the previous version.
        
        Args:
            service_name (str): Name of the service to rollback
            
        Returns:
            str: Rollback operation status and version information
        """
        prev_version = f"v1.{random.randint(10, 50)}.{random.randint(0, 9)}"
        return f"Rolling back {service_name} to previous version {prev_version}. Rollback initiated. Expected completion: 3-5 minutes. Health checks will resume automatically."
    
    def get_deployment_logs(self, pipeline_id: str, lines: int = 20) -> str:
        """
        Retrieve deployment logs from a CI/CD pipeline.
        
        Args:
            pipeline_id (str): The pipeline identifier
            lines (int): Number of log lines to retrieve (default: 20)
            
        Returns:
            str: Multi-line string containing deployment logs
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        build_num = random.randint(100, 999)
        return f"""[{timestamp}] BUILD #{build_num}: Starting deployment pipeline {pipeline_id}
[{timestamp}] BUILD #{build_num}: Code checkout completed from main branch
[{timestamp}] BUILD #{build_num}: Running unit tests... 42/42 tests passed
[{timestamp}] BUILD #{build_num}: Building Docker image: app:v1.2.3
[{timestamp}] BUILD #{build_num}: Pushing image to registry... Complete
[{timestamp}] BUILD #{build_num}: Deploying to staging environment... Success
[{timestamp}] BUILD #{build_num}: Running integration tests... All passed
[{timestamp}] BUILD #{build_num}: Deploying to production environment... Success"""
    
    def list_ci_cd_pipelines(self) -> list:
        """
        List all CI/CD pipelines and their current status.
        
        Returns:
            list: List of dictionaries containing pipeline information
        """
        return [
            {"pipeline_id": "web-app-deploy", "status": "SUCCESS", "last_run": "2024-01-23T10:30:00Z", "build_number": 127},
            {"pipeline_id": "api-service-deploy", "status": "RUNNING", "last_run": "2024-01-23T11:00:00Z", "build_number": 89},
            {"pipeline_id": "database-migration", "status": "FAILED", "last_run": "2024-01-23T09:15:00Z", "build_number": 45},
            {"pipeline_id": "mobile-app-build", "status": "SUCCESS", "last_run": "2024-01-23T08:45:00Z", "build_number": 203}
        ]
    
    def check_build_status(self, build_id: str) -> dict:
        """
        Check the status of a specific build.
        
        Args:
            build_id (str): The build identifier
            
        Returns:
            dict: Build status information including progress and timing
        """
        statuses = ["PENDING", "RUNNING", "SUCCESS", "FAILED"]
        status = random.choice(statuses)
        
        return {
            "build_id": build_id,
            "status": status,
            "started_at": (datetime.now() - timedelta(minutes=random.randint(5, 60))).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "duration": f"{random.randint(2, 15)} minutes" if status in ["SUCCESS", "FAILED"] else "In progress",
            "commit_sha": f"abc123{random.randint(1000, 9999)}",
            "branch": "main",
            "tests_passed": random.randint(35, 50),
            "tests_failed": random.randint(0, 3) if status != "SUCCESS" else 0
        }