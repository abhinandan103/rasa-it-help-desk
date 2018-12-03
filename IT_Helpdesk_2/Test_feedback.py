port1 = input('port_user')
port2 = input('port_agent')
from websocket import WebSocket
import json
ws_user = WebSocket()
ws_agent = WebSocket()
ws_user.connect('ws://192.168.0.129:'+port1+'/webhook')
ws_agent.connect('ws://192.168.0.129:'+port2+'/webhook')

def login(idd):
    return json.dumps({'status':'login','sender':idd})

def send_msg(idd,message):
    return json.dumps({'text':message,'sender':idd})


def send_feedback(idd,feedback):
    return json.dumps({'feedback':feedback,'sender':idd})

import numpy as np
sender_id = str(np.random.randint(123456))
ws_user.send(login(sender_id))
ws_agent.send(login('HA_agent5+'+sender_id))
print(ws_user.recv())
ws_agent.recv()
ws_user.send(send_msg(sender_id,'Hi'))
print(ws_user.recv())

ws_user.send(send_msg(sender_id,'I want to reset my omniview password'))
print(ws_user.recv())

ws_user.send(send_msg(sender_id,'john1234'))
print(ws_user.recv())
ws_user.send(send_msg(sender_id,'yes'))
print(ws_user.recv())
ws_user.send(send_msg(sender_id,'1234'))
print(ws_user.recv())
ws_user.send(send_msg(sender_id,'Ok'))
print(ws_user.recv())
ws_user.send(send_msg(sender_id,'No'))
a=ws_user.recv()
print(a)
a = json.loads(a)
ws_user.send(send_feedback(sender_id,{"chat_feedback":"5","event_id":a['event_id']}))
print(ws_user.recv())
