"""
Middleware for Rate Limiting, Circuit Breaker, and Request Tracking
"""
import time
import logging
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from collections import defaultdict
from datetime import datetime, timedelta
import asyncio

logger = logging.getLogger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware using sliding window"""
    
    def __init__(self, app, requests_per_minute: int = 100, burst: int = 200):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst = burst
        self.requests = defaultdict(list)
        self.cleanup_task = None
    
    async def dispatch(self, request: Request, call_next):
        # Get client identifier (IP address)
        client_ip = request.client.host
        
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        current_time = time.time()
        minute_ago = current_time - 60
        
        # Clean old requests
        self.requests[client_ip] = [
            req_time for req_time in self.requests[client_ip]
            if req_time > minute_ago
        ]
        
        # Check rate limit
        if len(self.requests[client_ip]) >= self.requests_per_minute:
            logger.warning("Rate limit exceeded for %s", client_ip)
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {self.requests_per_minute} requests per minute."
            )
        
        # Check burst limit
        recent_requests = [
            req_time for req_time in self.requests[client_ip]
            if req_time > current_time - 10  # Last 10 seconds
        ]
        
        if len(recent_requests) >= self.burst:
            logger.warning("Burst limit exceeded for %s", client_ip)
            raise HTTPException(
                status_code=429,
                detail=f"Burst limit exceeded. Maximum {self.burst} requests per 10 seconds."
            )
        
        # Add current request
        self.requests[client_ip].append(current_time)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - len(self.requests[client_ip])
        )
        response.headers["X-RateLimit-Reset"] = str(int(minute_ago + 60))
        
        return response


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """Circuit breaker to prevent cascading failures"""
    
    def __init__(self, app, failure_threshold: int = 5, timeout: int = 60):
        super().__init__(app)
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = defaultdict(int)
        self.last_failure_time = defaultdict(float)
        self.circuit_open = defaultdict(bool)
    
    async def dispatch(self, request: Request, call_next):
        endpoint = f"{request.method}:{request.url.path}"
        
        if self.circuit_open[endpoint]:
            if time.time() - self.last_failure_time[endpoint] > self.timeout:
                logger.info("Circuit breaker: Attempting to close circuit for %s", endpoint)
                self.circuit_open[endpoint] = False
                self.failures[endpoint] = 0
            else:
                logger.warning("Circuit breaker: Circuit open for %s", endpoint)
                raise HTTPException(
                    status_code=503,
                    detail="Service temporarily unavailable. Circuit breaker is open."
                )
        
        try:
            response = await call_next(request)
            
            # Reset failures on success
            if response.status_code < 500:
                self.failures[endpoint] = 0
            else:
                self._record_failure(endpoint)
            
            return response
            
        except (HTTPException, RuntimeError, ValueError) as e:
            self._record_failure(endpoint)
            raise
    
    def _record_failure(self, endpoint: str):
        """Record a failure and open circuit if threshold reached"""
        self.failures[endpoint] += 1
        self.last_failure_time[endpoint] = time.time()
        
        if self.failures[endpoint] >= self.failure_threshold:
            self.circuit_open[endpoint] = True
            logger.error(
                "Circuit breaker: Opening circuit for %s after %d failures",
                endpoint,
                self.failures[endpoint]
            )


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Track request metrics for monitoring"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = defaultdict(int)
        self.request_duration = defaultdict(list)
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Track metrics
            endpoint = f"{request.method}:{request.url.path}"
            self.request_count[endpoint] += 1
            self.request_duration[endpoint].append(duration)
            
            # Add timing header
            response.headers["X-Process-Time"] = f"{duration:.3f}s"
            
            # Log slow requests
            if duration > 5.0:
                logger.warning("Slow request: %s took %.2fs", endpoint, duration)

            return response

        except (RuntimeError, ValueError) as e:
            duration = time.time() - start_time
            logger.error(
                "Request failed: %s %s after %.2fs - %s",
                request.method,
                request.url.path,
                duration,
                str(e)
            )
            raise
    
    def get_metrics(self):
        """Get current metrics"""
        metrics = {}
        for endpoint, durations in self.request_duration.items():
            if durations:
                metrics[endpoint] = {
                    "count": self.request_count[endpoint],
                    "avg_duration": sum(durations) / len(durations),
                    "max_duration": max(durations),
                    "min_duration": min(durations)
                }
        return metrics

