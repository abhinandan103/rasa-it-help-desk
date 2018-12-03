from websocket import WebSocket
import json
import pytest
import numpy as np
import numpy as np
import requests
import simplejson
import io

global env_config
with io.open("test_envconfig.json", encoding="utf-8") as f:
    env_config = f.read()
    env_config = simplejson.loads(env_config)

def login(idd):
    return json.dumps({'status':'login','sender':idd})

def send_msg(idd,message):
    return json.dumps({'text':message,'sender':idd})

def test_websocket_connection():


    ws_user = WebSocket()
    ws_agent = WebSocket()
    sender_id = str(np.random.randint(1234562523))+"_"+"john123"

    ws_user.connect(env_config['RASA_URL'])
    ws_agent.connect(env_config['RASA_URL'])
    ws_user.send(login(sender_id))
    received_data=ws_user.recv()


    ws_user.send(send_msg(sender_id,'Hi'))
    received_data=ws_user.recv()


    ws_user.send(send_msg(sender_id,'Yes'))
    received_data=ws_user.recv()


    ws_agent.send(login('HA_agent5+'+sender_id))
    received_data=ws_agent.recv()


    ws_user.send(send_msg(sender_id,'Yes'))
    received_data=ws_user.recv()

    ws_user.send(send_msg(sender_id,'ok'))
    received_data = requests.get(env_config['conv_listen_address'] + "?sender_id=" + sender_id)

    ws_agent.send(send_msg('HA_agent5+'+sender_id, "Hi john, Your payoff amount is $1461"))
    received_data=ws_user.recv()
    ws_user.send(send_msg(sender_id,"ok"))

    return 'success'



if __name__ == '__main__':
    print(test_websocket_connection())
    
