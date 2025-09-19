import json
from datetime import datetime, timedelta
import random

class NetworkingTools:
    """DevOps tools for managing networking, DNS, load balancing, and connectivity"""
    
    def check_vpc_connectivity(self, vpc_id: str) -> dict:
        """
        Check connectivity status within a VPC.
        
        Args:
            vpc_id (str): The VPC identifier
            
        Returns:
            dict: VPC connectivity information including subnets and routing
        """
        return {
            "vpc_id": vpc_id,
            "state": "available",
            "cidr_block": "10.0.0.0/16",
            "subnets": [
                {"subnet_id": "subnet-abc123", "cidr": "10.0.1.0/24", "az": "us-east-1a", "available_ips": random.randint(200, 250)},
                {"subnet_id": "subnet-def456", "cidr": "10.0.2.0/24", "az": "us-east-1b", "available_ips": random.randint(200, 250)}
            ],
            "route_tables": 3,
            "internet_gateway": "igw-xyz789",
            "nat_gateways": 2,
            "connectivity_status": "healthy"
        }
    
    def list_security_groups(self) -> list:
        """
        List all security groups in the current VPC.
        
        Returns:
            list: List of dictionaries containing security group information
        """
        return [
            {"group_id": "sg-web123456", "name": "web-servers", "description": "HTTP/HTTPS access", "vpc_id": "vpc-abc123"},
            {"group_id": "sg-db789012", "name": "database", "description": "Database access from app tier", "vpc_id": "vpc-abc123"},
            {"group_id": "sg-lb345678", "name": "load-balancer", "description": "Public load balancer", "vpc_id": "vpc-abc123"},
            {"group_id": "sg-ssh901234", "name": "ssh-access", "description": "SSH access for administrators", "vpc_id": "vpc-abc123"}
        ]
    
    def get_firewall_rules(self, firewall_id: str) -> list:
        """
        Get firewall rules for a specific firewall.
        
        Args:
            firewall_id (str): The firewall identifier
            
        Returns:
            list: List of firewall rules with protocols, ports, and sources
        """
        return [
            {"rule_id": 1, "protocol": "TCP", "port": "80", "source": "0.0.0.0/0", "action": "ALLOW", "description": "HTTP traffic"},
            {"rule_id": 2, "protocol": "TCP", "port": "443", "source": "0.0.0.0/0", "action": "ALLOW", "description": "HTTPS traffic"},
            {"rule_id": 3, "protocol": "TCP", "port": "22", "source": "10.0.0.0/8", "action": "ALLOW", "description": "SSH from private networks"},
            {"rule_id": 4, "protocol": "TCP", "port": "3306", "source": "sg-web123456", "action": "ALLOW", "description": "MySQL from web servers"},
            {"rule_id": 5, "protocol": "ALL", "port": "*", "source": "*", "action": "DENY", "description": "Default deny all"}
        ]
    
    def update_dns_record(self, domain: str, record: str, value: str) -> str:
        """
        Update a DNS record for a domain.
        
        Args:
            domain (str): The domain name
            record (str): The record type (A, CNAME, MX, etc.)
            value (str): The new record value
            
        Returns:
            str: Confirmation message of DNS update operation
        """
        return f"DNS record updated: {record} record for {domain} set to {value}. TTL: 300 seconds. Propagation time: 5-15 minutes globally."
    
    def check_load_balancer_routing(self, lb_name: str) -> dict:
        """
        Check load balancer routing configuration and target health.
        
        Args:
            lb_name (str): The load balancer name
            
        Returns:
            dict: Load balancer routing information and target status
        """
        return {
            "lb_name": lb_name,
            "type": "Application Load Balancer",
            "dns_name": f"{lb_name}-{random.randint(1000000, 9999999)}.us-east-1.elb.amazonaws.com",
            "listeners": [
                {"port": 80, "protocol": "HTTP", "default_action": "redirect to HTTPS"},
                {"port": 443, "protocol": "HTTPS", "default_action": "forward to target group"}
            ],
            "target_groups": [
                {"name": "web-servers", "healthy_targets": 3, "unhealthy_targets": 0, "total_targets": 3},
                {"name": "api-servers", "healthy_targets": 2, "unhealthy_targets": 1, "total_targets": 3}
            ],
            "routing_algorithm": "round_robin"
        }
    
    def dns_lookup(self, domain: str) -> dict:
        """
        Perform DNS lookup for a domain.
        
        Args:
            domain (str): The domain name to lookup
            
        Returns:
            dict: DNS lookup results with various record types
        """
        return {
            "domain": domain,
            "query_time": f"{random.randint(10, 150)} ms",
            "records": {
                "A": [f"52.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}", f"54.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"],
                "AAAA": ["2001:db8::1", "2001:db8::2"],
                "CNAME": f"alias.{domain}",
                "MX": [f"10 mail.{domain}", f"20 mail2.{domain}"],
                "TXT": [f'"v=spf1 include:_spf.{domain} ~all"', '"google-site-verification=abc123"']
            },
            "authoritative_servers": [f"ns1.{domain}", f"ns2.{domain}"]
        }
    
    def get_load_balancer_health(self, lb_name: str) -> dict:
        """
        Get health status of a load balancer and its targets.
        
        Args:
            lb_name (str): The load balancer name
            
        Returns:
            dict: Load balancer health information and metrics
        """
        return {
            "lb_name": lb_name,
            "state": "active",
            "health_status": "healthy",
            "active_connections": random.randint(50, 500),
            "new_connections_per_second": random.randint(10, 100),
            "target_response_time": f"{random.randint(50, 200)} ms",
            "healthy_hosts": random.randint(2, 5),
            "unhealthy_hosts": random.randint(0, 1),
            "error_rate": f"{random.uniform(0.1, 2.0):.1f}%",
            "last_health_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def check_service_health(self, service_name: str) -> dict:
        """
        Check the health status of a service.
        
        Args:
            service_name (str): Name of the service to check
            
        Returns:
            dict: Service health information including endpoints and status
        """
        return {
            "service_name": service_name,
            "status": random.choice(["healthy", "degraded", "unhealthy"]),
            "response_time": f"{random.randint(50, 300)} ms",
            "uptime": f"{random.randint(1, 30)} days, {random.randint(1, 23)} hours",
            "endpoints": [
                {"url": f"https://{service_name}.company.com/health", "status": 200, "response_time": f"{random.randint(20, 100)} ms"},
                {"url": f"https://{service_name}.company.com/metrics", "status": 200, "response_time": f"{random.randint(30, 120)} ms"}
            ],
            "dependencies": ["database", "redis", "auth-service"],
            "last_check": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def ping_host(self, host: str) -> str:
        """
        Ping a host to check network connectivity.
        
        Args:
            host (str): The hostname or IP address to ping
            
        Returns:
            str: Ping results with latency and packet loss information
        """
        return f"""PING {host} (52.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}): 56 data bytes
64 bytes from {host}: icmp_seq=1 ttl=54 time={random.randint(10,50)}.{random.randint(1,9)} ms
64 bytes from {host}: icmp_seq=2 ttl=54 time={random.randint(10,50)}.{random.randint(1,9)} ms
64 bytes from {host}: icmp_seq=3 ttl=54 time={random.randint(10,50)}.{random.randint(1,9)} ms
64 bytes from {host}: icmp_seq=4 ttl=54 time={random.randint(10,50)}.{random.randint(1,9)} ms

--- {host} ping statistics ---
4 packets transmitted, 4 received, 0% packet loss
round-trip min/avg/max/stddev = {random.randint(15,25)}.{random.randint(1,9)}/{random.randint(25,35)}.{random.randint(1,9)}/{random.randint(35,45)}.{random.randint(1,9)}/{random.randint(5,15)}.{random.randint(1,9)} ms"""
    
    def get_ssl_certificate_info(self, domain: str) -> dict:
        """
        Get SSL certificate information for a domain.
        
        Args:
            domain (str): The domain name to check
            
        Returns:
            dict: SSL certificate details including validity and issuer information
        """
        issue_date = datetime.now() - timedelta(days=random.randint(30, 300))
        expiry_date = issue_date + timedelta(days=365)
        
        return {
            "domain": domain,
            "certificate_status": "valid",
            "issuer": "Let's Encrypt Authority X3",
            "subject": f"CN={domain}",
            "serial_number": f"{random.randint(100000000, 999999999):x}",
            "issued_date": issue_date.strftime("%Y-%m-%d"),
            "expiry_date": expiry_date.strftime("%Y-%m-%d"),
            "days_until_expiry": (expiry_date - datetime.now()).days,
            "signature_algorithm": "SHA256withRSA",
            "key_size": "2048 bits",
            "san_domains": [f"www.{domain}", f"api.{domain}", f"cdn.{domain}"]
        }