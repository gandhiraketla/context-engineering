import json
from datetime import datetime, timedelta
import random

class EC2AndKubernetesTools:
    """DevOps tools for managing EC2 instances and Kubernetes clusters"""
    
    def get_ec2_status(self, instance_id: str) -> dict:
        """
        Get the current status of an EC2 instance.
        
        Args:
            instance_id (str): The EC2 instance ID
            
        Returns:
            dict: Instance status information including state, uptime, and resource usage
        """
        return {
            "instance_id": instance_id,
            "state": "running",
            "uptime": f"{random.randint(1, 30)} days, {random.randint(1, 23)} hours",
            "instance_type": "t3.medium",
            "cpu_utilization": f"{random.randint(15, 85)}%",
            "memory_usage": f"{random.randint(45, 90)}%",
            "disk_usage": f"{random.randint(25, 75)}%",
            "availability_zone": "us-east-1a",
            "public_ip": f"52.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
        }
    
    def get_ec2_logs(self, instance_id: str, lines: int = 20) -> str:
        """
        Retrieve system logs from an EC2 instance.
        
        Args:
            instance_id (str): The EC2 instance ID
            lines (int): Number of log lines to retrieve (default: 20)
            
        Returns:
            str: Multi-line string containing system logs
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"""[{timestamp}] INFO: System boot completed successfully
[{timestamp}] INFO: Apache HTTP Server started on port 80
[{timestamp}] WARNING: High memory usage detected (87%)
[{timestamp}] INFO: Cron job completed: daily backup
[{timestamp}] INFO: Security updates installed: 3 packages
[{timestamp}] INFO: Load balancer health check passed"""
    
    def restart_ec2_instance(self, instance_id: str) -> str:
        """
        Restart an EC2 instance.
        
        Args:
            instance_id (str): The EC2 instance ID to restart
            
        Returns:
            str: Confirmation message of restart operation
        """
        return f"Instance {instance_id} restart initiated. Expected downtime: 2-3 minutes. Current status: stopping -> pending -> running"
    
    def scale_ec2_instance(self, instance_id: str, new_type: str) -> str:
        """
        Scale an EC2 instance to a different instance type.
        
        Args:
            instance_id (str): The EC2 instance ID
            new_type (str): The new instance type (e.g., 't3.large')
            
        Returns:
            str: Status message of scaling operation
        """
        return f"Scaling instance {instance_id} to {new_type}. Operation will complete in 5-10 minutes. Instance will be temporarily stopped during resize."
    
    def list_ec2_instances(self, filter_state: str = None) -> list:
        """
        List all EC2 instances in the current region.
        
        Returns:
            list: List of dictionaries containing instance information
        """
        return [
            {"instance_id": "i-0abc123def456789", "name": "web-server-prod", "state": "running", "type": "t3.medium"},
            {"instance_id": "i-0def456ghi789abc", "name": "db-server-prod", "state": "running", "type": "r5.large"},
            {"instance_id": "i-0ghi789jkl012def", "name": "worker-node-1", "state": "stopped", "type": "c5.xlarge"},
            {"instance_id": "i-0jkl012mno345ghi", "name": "staging-app", "state": "running", "type": "t3.small"}
        ]
    
    def get_k8s_pod_status(self, pod_name: str, namespace: str) -> dict:
        """
        Get the status of a Kubernetes pod.
        
        Args:
            pod_name (str): Name of the Kubernetes pod
            namespace (str): Kubernetes namespace
            
        Returns:
            dict: Pod status information including phase, restarts, and resource usage
        """
        return {
            "pod_name": pod_name,
            "namespace": namespace,
            "phase": "Running",
            "ready": "1/1",
            "restarts": random.randint(0, 5),
            "age": f"{random.randint(1, 15)}d",
            "cpu_usage": f"{random.randint(10, 80)}m",
            "memory_usage": f"{random.randint(128, 512)}Mi",
            "node": f"node-{random.randint(1,3)}.cluster.local"
        }
    
    def get_k8s_pod_logs(self, pod_name: str, namespace: str, lines: int = 20) -> str:
        """
        Retrieve logs from a Kubernetes pod.
        
        Args:
            pod_name (str): Name of the Kubernetes pod
            namespace (str): Kubernetes namespace
            lines (int): Number of log lines to retrieve (default: 20)
            
        Returns:
            str: Multi-line string containing pod logs
        """
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        return f"""[{timestamp}] INFO: Starting application server
[{timestamp}] INFO: Connected to database successfully
[{timestamp}] INFO: Health check endpoint responding on :8080/health
[{timestamp}] WARNING: Slow query detected: SELECT * FROM users (2.3s)
[{timestamp}] INFO: Processing 15 requests in queue
[{timestamp}] INFO: Cache hit ratio: 94.2%"""
    
    def restart_k8s_pod(self, pod_name: str, namespace: str) -> str:
        """
        Restart a Kubernetes pod by deleting it (will be recreated by deployment).
        
        Args:
            pod_name (str): Name of the Kubernetes pod
            namespace (str): Kubernetes namespace
            
        Returns:
            str: Confirmation message of restart operation
        """
        return f"Pod {pod_name} in namespace {namespace} deleted. New pod will be created automatically by deployment controller. Expected restart time: 30-60 seconds."
    
    def scale_k8s_deployment(self, deployment: str, namespace: str, replicas: int) -> str:
        """
        Scale a Kubernetes deployment to the specified number of replicas.
        
        Args:
            deployment (str): Name of the Kubernetes deployment
            namespace (str): Kubernetes namespace
            replicas (int): Desired number of replicas
            
        Returns:
            str: Status message of scaling operation
        """
        return f"Deployment {deployment} in namespace {namespace} scaled to {replicas} replicas. Current: 2/2 ready, Target: {replicas}/{replicas}"
    
    def list_k8s_pods(self, namespace: str) -> list:
        """
        List all pods in a Kubernetes namespace.
        
        Args:
            namespace (str): Kubernetes namespace
            
        Returns:
            list: List of dictionaries containing pod information
        """
        return [
            {"name": "frontend-app-7d4b9c8f6-x2m9p", "ready": "1/1", "status": "Running", "restarts": 0, "age": "5d"},
            {"name": "backend-api-5f8a7b2c3-y4n8q", "ready": "1/1", "status": "Running", "restarts": 1, "age": "3d"},
            {"name": "database-6c9e4d1f2-z5p7r", "ready": "1/1", "status": "Running", "restarts": 0, "age": "12d"},
            {"name": "worker-8a3f6e9b4-w1q3t", "ready": "1/1", "status": "Running", "restarts": 2, "age": "7d"}
        ]