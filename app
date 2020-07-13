server {
 server_name _;
 listen 80 default_server;
 
 location / {
 proxy_pass http://127.0.0.1:5000;
 proxy_set_header Host $host;
 proxy_set_header X-Real-IP ip_address;
 }
}
