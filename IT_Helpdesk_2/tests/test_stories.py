import sys, os
sys.path.append(os.path.abspath(os.path.join('..'))) # Need to find another way to do this


import simplejson
import io
from websocket import WebSocket
import _pickle
import redis
import numpy as np
import json
import time
import requests

from suprath_nlu.wrapper import Interpreter
from idare import KerasIDarePolicy as KP
from rasa_core.featurizers import FloatSingleStateFeaturizer,MaxHistoryTrackerFeaturizer
from rasa_core.domain import OntologyDomain
from rasa_core.trackers import DialogueStateTracker
from rasa_core.events import UserUttered,ActionExecuted



global env_config

with io.open("test_envconfig.json", encoding="utf-8") as f:
    env_config = f.read()
    env_config = simplejson.loads(env_config)
global ws_user
global ws_agent
ws_user = WebSocket()
ws_agent = WebSocket()
global sender_id
def login(idd):
    return json.dumps({'status':'login','sender':idd})

def send_msg(idd,message):
    return json.dumps({'text':message,'sender':idd})


def send_feedback(idd,feedback):
    return json.dumps({'feedback':feedback,'sender':idd})


def get_interpreter_result(interp_data):
    return interp_data['act']['name'],interp_data['slots'][0]['entity']

def get_user_tracker(user_id):
    red = redis.StrictRedis()
    dialogue =  _pickle.loads(red.get(user_id))
    domain = OntologyDomain.load(env_config['domain'])
    tracker =  DialogueStateTracker(sender_id = dialogue.name,
                                    sys_goals = domain.sys_goals,
                                    usr_slots = domain.usr_slots,
                                    methods   = domain.methods,
                                    processes = domain.processes)

    tracker.recreate_from_dialogue(dialogue)
    return tracker

def send_and_assert_nlu_act(message,act,slots=None, typ='user'):
    if typ=='agent': 
        ws = ws_agent
        idd = agent_id
    else:
        ws = ws_user
        idd = sender_id
    ws.send(send_msg(idd,message))
    result = json.loads(ws.recv())
    time.sleep(0.2)
    tracker = get_user_tracker(sender_id)
    latest_user_message = None
    for i in range(len(tracker.events)-1,-1,-1):
        event = tracker.events[i]
        if isinstance(event,UserUttered) and typ=='user' or isinstance(event,ActionExecuted) and typ=='agent':
            latest_user_message = event
            break
    assert latest_user_message is not None, "Latest user message unavailable"
    if typ=='user':
        assert latest_user_message.nlu_act['name'] == act and latest_user_message.nlu_act['confidence'] > 0.4 ,\
                f"nlu act should be {act}, prediction is: {latest_user_message.nlu_act['name']} with confidence {latest_user_message.nlu_act['confidence']}\
                  for query {message}"
    else:
        assert latest_user_message.action_act['name'] == act and latest_user_message.action_act['confidence'] > 0.4,\
                f"action act should be {act}, prediction is: {latest_user_message.action_act['name']} with confidence {latest_user_message.action_act['confidence']}\
                 for query {message}"

    if slots==None:
        return result
    for check_slot in slots:
        SLOTS = latest_user_message.nlu_slots if typ=='user' else latest_user_message.action_slots
        for slot in SLOTS:
            if slot['slot'] == check_slot:
                assert slot['confidence'] > 0.4, (f'confidence of slot {check_slot} is less than threshold.'
                                                  f' It is {slot["confidence"]}. Slot is {slot}')
                if slots[check_slot] is not None:
                    assert slot['value'] == slots[check_slot], (f"f{check_slot} NLU slot should be "   
                                                                f" {slots[check_slot]} but got "
                                                                f"{slot['value']}")

    return result

def assert_nlu_act(message,act,slots=None):
    res = requests.get('http://'+env_config['server_ip']+':'+env_config['server_port']+"/nlu_parse",params={'q':message})
    assert res.status_code == 200
    result = res.json()
    
    assert result['act']['name'] == act and result['act']['confidence'] > 0.4 ,\
                f"nlu act should be {act}, prediction is: {result['act']['name']} with confidence {result['act']['confidence']}\
                  for query {message}"

    if slots==None:
        return

    for check_slot in slots:
        SLOTS = result['slots']
        for slot in SLOTS:
            if slot['slot'] == check_slot:
                assert slot['confidence'] > 0.4, (f'confidence of slot {check_slot} is less than threshold.'
                                                  f' It is {slot["confidence"]}. Slot is {slot}')
                if slots[check_slot] is not None:
                    assert slot['value'] == slots[check_slot], (f"f{check_slot} NLU slot should be "   
                                                                f" {slots[check_slot]} but got "
                                                                f"{slot['value']}")

def get_ha_id():
    tracker = get_user_tracker(sender_id)
    try:
        ha_id = tracker.events[-1].ha_id
    except AttributeError:
        raise ('ha id is absent')
    return ha_id

def assert_mode(mode):

    tracker = get_user_tracker(sender_id)
    assert tracker.mode == mode, f'mode is {tracker.mode} instead of mode'

def assert_belief_state(BELIEF_STATE):

    tracker = get_user_tracker(sender_id)
    belief_state = tracker.belief_state
    if 'processes' in BELIEF_STATE:
        proc = None
        max_val = -1
        for proc,details in belief_state['processes'].items():
            if details['confidence'] > max_val:
                max_val = details['confidence']
                name = proc
        assert name == BELIEF_STATE['processes'], f"Process is {proc} instead of {BELIEF_STATE['processes']}" 
   
    if 'methods' in BELIEF_STATE:
        method = None
        max_val = -1
        for method,details in belief_state['methods'].items():
            if details['confidence'] > max_val:
                max_val = details['confidence']
                name = method
        assert name == BELIEF_STATE['methods'], f"Method is {method} instead of {BELIEF_STATE['methods']}" 

    if 'usr_slots' in BELIEF_STATE:
        for key,value in BELIEF_STATE['usr_slots'].items():
            pred = belief_state['usr_slots'][key]['value']
            assert value is None or pred == value, f"user slot {key} should have {value} value but is {pred}"
            assert belief_state['usr_slots'][key]['confidence'] > 0.4, f"usr slot {key} confidence is low for {value}"


    if 'sys_goals' in BELIEF_STATE:
        for key,value in BELIEF_STATE['sys_goals'].items():
            pred = belief_state['sys_goals'][key]['value']
            assert value is None or pred == value, f"sys goal {key} should have {value} value but is {pred}"
            assert belief_state['sys_goals'][key]['confidence'] > 0.4, f"sys goal {key} confidence is low for {value}"

def test_story_1():
    global sender_id
    global agent_id
    sender_id = str(np.random.randint(1232256))
    agent_id = 'HA_agent3+'+sender_id
    print(f'sender id is {sender_id}')
    ws_user.connect('ws://'+env_config['server_ip']+':'+env_config['userchannelport']+'/webhook')
    ws_agent.connect('ws://'+env_config['server_ip']+':'+env_config['hachannelport']+'/webhook')


    ws_user.send(login(sender_id))
    ws_agent.send(login(agent_id))

    ws_user.recv()
    ws_agent.recv()

    send_and_assert_nlu_act('Hi','greet')

    send_and_assert_nlu_act('I want to reset my password','inform',{'password_reset':None})

    assert_belief_state({'processes':'password_reset',
                         'sys_goals':{'requested_slot':'ID'}})

    send_and_assert_nlu_act('john1234','inform',{'ID':'john1234'})

    assert_belief_state({'usr_slots':{'ID':'john1234'}})

    send_and_assert_nlu_act('john1234','inform',{'ID':'john1234'})
   
    assert_belief_state({'sys_goals':{'requested_slot':'send_otp'},
                         'usr_slots':{'ID':'john1234'}})

    send_and_assert_nlu_act('yes','affirm')

    assert_belief_state({'sys_goals':{'requested_slot':'security_code'},
                         'usr_slots':{'send_otp':None}})

    send_and_assert_nlu_act('1234','inform',{'number':'1234'})

    assert_belief_state({'usr_slots':{'security_code':'1234'},
                         'methods': 'finished'})

    send_and_assert_nlu_act('thanks','gratitude')

    send_and_assert_nlu_act('no','deny')

    assert_belief_state({'methods': 'bye'})
    
    ws_user.close()
    ws_agent.close()

    return "Successfully tested first story"

def test_story_2():
    for i in range(3):
        global sender_id
        global agent_id
        sender_id = str(np.random.randint(1232256))
        print(f'sender id is {sender_id}')
        ws_user.connect('ws://'+env_config['server_ip']+':'+env_config['userchannelport']+'/webhook')
        ws_agent.connect('ws://'+env_config['server_ip']+':'+env_config['hachannelport']+'/webhook')


        ws_user.send(login(sender_id))

        ws_user.recv()

        send_and_assert_nlu_act('Hi','greet')

        send_and_assert_nlu_act('I want access to this machine: app1','request',{'access':None,'machine':'app1'})

        assert_mode('agent_assist')

        ha_id = get_ha_id()
        print(ha_id)
        agent_id = ha_id+'+'+sender_id
        ws_agent.send(login(agent_id))
        ws_agent.recv()

        send_and_assert_nlu_act(KP.NLG_top3(None,'request',['approval'])[i],'confirm',{'approval':None},typ='agent')

        assert_belief_state({'sys_goals':{'requested_slot':'approval'}})
        
        send_and_assert_nlu_act('yes I have','affirm')

        try:
            send_and_assert_nlu_act(KP.NLG_top3(None,'request',['request_number'])[i],'request',{'request_number':None},typ='agent')
        except AssertionError:
            send_and_assert_nlu_act(KP.NLG_top3(None,'request',['request_number'])[i],'confirm',{'request_number':None},typ='agent')
    

        send_and_assert_nlu_act('It is 1234','inform',{'number':'1234'})

        assert_belief_state({'usr_slots':{'request_number':'1234'}})
        
        send_and_assert_nlu_act(KP.NLG_top3(None,'inform',['access_grant','finished'])[i],'inform',{'access_grant':None,
                                                                    'finished':None}, typ='agent')
        
        assert_belief_state({'methods':'finished'})

        send_and_assert_nlu_act(KP.NLG_top3(None,'confirm',['login_able'])[i],'confirm',{'login_able':None},typ='agent')

        assert_belief_state({'sys_goals':{'requested_slot':'login_able'}})

        send_and_assert_nlu_act('yes I am','affirm')

        assert_belief_state({'usr_slots':{'login_able':None}})

        send_and_assert_nlu_act(KP.NLG_top3(None,'anythingelse',[])[i],'anythingelse',typ='agent')

        send_and_assert_nlu_act('Do I have admin access','confirm',{'admin_access':None})

        assert_belief_state({'usr_slots':{'admin_access':None}})

        send_and_assert_nlu_act(KP.NLG_top3(None,'deny',['admin_access'])[i],'deny',typ='agent')

        send_and_assert_nlu_act('ok','affirm')

        send_and_assert_nlu_act(KP.NLG_top3(None,'anythingelse',[])[i],'anythingelse',typ='agent')

        send_and_assert_nlu_act('no','deny')

        assert_belief_state({'methods': 'bye'})

        ws_user.close()
        ws_agent.close()



    return "Successfully tested second story"


def test_story_3():
    """Story of excel to pdf conversion"""
    global sender_id
    global agent_id
    sender_id = str(np.random.randint(1232256))
    agent_id = 'HA_agent3+'+sender_id
    print(f'sender id is {sender_id}')
    ws_user.connect('ws://'+env_config['server_ip']+':'+env_config['userchannelport']+'/webhook')
    ws_agent.connect('ws://'+env_config['server_ip']+':'+env_config['hachannelport']+'/webhook')


    ws_user.send(login(sender_id))
    ws_agent.send(login(agent_id))

    ws_user.recv()
    ws_agent.recv()

    result = send_and_assert_nlu_act('Hi','greet')
    assert_nlu_act(result['text'],'greet')

    result = send_and_assert_nlu_act('I am unable to convert excel files to PDF.','inform',{'issue':None,
                                                                                    'convert':None,
                                                                                    'xls2pdf':None})

    assert_nlu_act(result['text'],'request',{'pdf_converter':None})

    assert_belief_state({'processes':'Software Troubleshooting',
                        'methods':'byname',
                         'usr_slots':{'issue':None,'convert':None,'xls2pdf':None},
                         'sys_goals':{'requested_slot':'pdf_converter'}})

    result = send_and_assert_nlu_act('I have adobe acrobat pro installed.','inform',{'pdf_converter':'adobe acrobat pro'})
    assert_nlu_act(result['text'],'request',{'pdf_converter':None,'login':None})

    assert_belief_state({'usr_slots':{'pdf_converter':'adobe acrobat pro'},
                         'sys_goals':{'requested_slot':'login'}})

    result = send_and_assert_nlu_act('Oh do I need to do that? How can I login?','request',{'howto':None,'login':None})
    assert_nlu_act(result['text'],'instruction')
   
    assert_belief_state({'sys_goals':{'instruction':None},
                          'usr_slots':{'login':None,'howto':None},})

    result = send_and_assert_nlu_act('Hmm let me try that.','wait')
    assert_nlu_act(result['text'],'confirm',{'login_able':None})

    assert_belief_state({'sys_goals':{'requested_slot':'login_able'},
                         'usr_slots':{'wait':None}})

    result = send_and_assert_nlu_act('Yeah, I logged in.','affirm')
    assert_nlu_act(result['text'],'confirm',{'resolved':None})

    assert_belief_state({'usr_slots':{'login':None},
                         'sys_goals':{'requested_slot':'resolved'}})


    result = send_and_assert_nlu_act('No I am not','deny')
    assert_nlu_act(result['text'],'instruction')

    assert_belief_state({'sys_goals':{'instruction':None}})

    result = send_and_assert_nlu_act('Oh, I didn’t realize that. Thank you.','wait')
    assert_nlu_act(result['text'],'confirm',{'resolved':None})

    assert_belief_state({'sys_goals':{'requested_slot':'resolved'}})

    result = send_and_assert_nlu_act('Yes it has! Thanks.','affirm')
    assert_nlu_act(result['text'],'anythingelse')

    assert_belief_state({'methods': 'finished'})

    result = send_and_assert_nlu_act('No, thanks!','deny')

    assert_belief_state({'methods': 'bye'})

    ws_user.close()
    ws_agent.close()

    
    ws_user.close()
    ws_agent.close()

    return "Successfully tested third story"


if __name__ == '__main__':
    print(test_story_3())
    print(test_story_2())
    print(test_story_1())

