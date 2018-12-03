
from flask import Flask, request
app = Flask(__name__)



from gevent.pywsgi import WSGIServer
http_port = 6200
http_server = WSGIServer(('0.0.0.0', http_port), app)
http_server.serve_forever()
