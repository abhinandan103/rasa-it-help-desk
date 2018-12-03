import requests
import logging
import os
from base64 import b64encode
from simplejson.errors import JSONDecodeError
from urllib3.exceptions import MaxRetryError
from requests.exceptions import ConnectionError,Timeout
import pymysql
import urllib.parse as urlparser
import hashlib

from rasa_core.exceptions import APIError
from rasa_core.utils import  read_json_file

logger = logging.getLogger(__name__)

try:
    path = './'
    configdata=read_json_file(os.path.join(path, "envconfig.json"))

except FileNotFoundError:
    path = '../'
    configdata=read_json_file(os.path.join(path, "envconfig.json"))
    #This exception is handled coz of one of the test cases

def get_sentiment(usermessage,conversationid):
    apitimeout=configdata['sentiment_api_timeout']
    if(configdata['sentiment_api_timeout']==None):
        apitimeout=30.0
    else:
        apitimeout=float(apitimeout)
    try:
        if configdata['sentiment_api'] !="":
            r=requests.post(configdata['sentiment_api'],json={"text":usermessage},timeout=apitimeout)
        else:
            return 0
    except(ConnectionError,MaxRetryError,ConnectionRefusedError,Timeout) as e:
        raise APIError(str(e))
    if r.status_code!=200:
        raise APIError
    try:
        jsr=r.json()
    except JSONDecodeError as e:
        raise APIError(str(e))

    if jsr.get('sentiment'):
        try:
            sentiment=float(jsr.get('sentiment'))
            connection=getDBConnection()
            try:
                with connection.cursor() as cursor:
                    sql = """insert into t_sentimenttrend (conversationid,timeofsentiment,sentimentscore) values (%s,current_timestamp(),%s)"""
                    cursor.execute(sql, (conversationid,sentiment))
                connection.commit()
            finally:
                connection.close()
        except ValueError as e:
            raise APIError
        return sentiment
    else:
        raise APIError

    raise APIError



def getDBConnection():
    connection = pymysql.connect(host=configdata['db_host'],
                                user=configdata['db_username'],
                                password=configdata['db_password'],
                                db='agentdb',
                                charset='utf8mb4',
                                cursorclass=pymysql.cursors.DictCursor)
    return connection

def predictAgent(message):
    result=''
    connection = getDBConnection()
    try:
        with connection.cursor() as cursor:
    # Read a single record
            sql = """select log.agent_name ,count(t1.agent_name)  activechatcount from t_agentlog log
        		left join t_agentconversations t1	on log.agent_name=t1.agent_name    
                    where log.role='agent'  group by log.agent_name order by activechatcount asc
                """
            cursor.execute(sql)
            result = cursor.fetchone()['agent_name']
            print(result)

        with connection.cursor() as cursor:
            sql="""insert into t_agentconversations (agent_name,conversationid,status,userid,intent,confidence,
            text,chatopentime) values (%s,%s,%s,%s,%s,%s,%s,%s)"""
            confidence = float(message['confidence']) * 100
            ts=message['chat_open_time']

            import datetime
            timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            cursor.execute(sql,(result,message['sender_id'],'assigned',message['ticket_details'].get('ID','unknown')
                            ,message['intent'],confidence,message['text'],timestamp))
        connection.commit()
    finally:
        connection.close()

    return 'HA_'+result


def smart_routing(idare,message,actions,latest_intent,confidences,ha_id,mode,tracker):
    if mode == 'virtual_agent':
        if actions == None or actions == [] or 'access' in message['intent']:
            agent_name=predictAgent(message);	
            #REDIRECTING
            return [('redirect',[])],agent_name,'agent_assist'
        else:
            return actions,None,'virtual_agent'
    if mode == 'agent_assist':
        if actions == None or actions == []:
            return None,ha_id,mode
        return None,ha_id,mode

    return actions,None,'virtual_agent'

def redirect_url(tracker):
    thres = 0.4
    email_id = ""
    if tracker.belief_state['usr_slots']['email_id'].get('confidence') > thres:
        email_id = tracker.belief_state['usr_slots']['email_id'].get('value')
    convid = tracker.sender_id
    string = (convid+email_id+'/').encode('utf-8')+b64encode(os.urandom(16))
    token = hashlib.sha224(string).hexdigest()

    connection = getDBConnection()
    try:
        with connection.cursor() as cursor:
            sql="""insert into t_clientportal_authorize (token,convid,usrid) values (%s,%s,%s)"""

            cursor.execute(sql,(token,convid,email_id))
        connection.commit()
    except Exception as e:
        logger.error(e)
        return None
    finally:
        connection.close()

    query = f"/main?token={token}"
    url = urlparser.urljoin(configdata['clientportal'],query)
    return url



def information_retrieval(ix):
    links = []
    if (ix==1):
        links.append(configdata['static_server']+'/static/pdf/local_machine.pdf')
        links.append(configdata['static_server']+'/static/pdf/account_activation_process.pdf')
        links.append(configdata['static_server']+'/static/pdf/it_faq.pdf')
        return links
    if (ix==0):
        links.append(configdata['static_server']+'/static/images/img1.png')
        links.append(configdata['static_server']+'/static/pdf/remote.pdf')
        links.append(configdata['static_server']+'/static/pdf/it_faq.pdf')
        return links
    if (ix==3):
        links.append(configdata['static_server']+'/static/images/adobe.png')
        links.append(configdata['static_server']+'/static/pdf/remote.pdf')
        links.append(configdata['static_server']+'/static/pdf/it_faq.pdf')
        return links
    return [""]*3






def decision_making(belief_state):
    connection = getDBConnection()
    user = belief_state['usr_slots']['ID'].get('value')
    if user is None:
        return [""],[""]
    THRES = 0.6
    context_info = None
    payoffbalance = None

    try:
        with connection.cursor() as cursor:
            # Read a single record
            sql = "select * from t_employeeinfo where id= %s "
            rows = cursor.execute(sql, (user))
            if (rows > 0):
                context_info = cursor.fetchone()
            else:
                logger.error(f'No entry in the table for decision making for id {user}')
                return [""],[""]
    finally:
        connection.close()

    info_fields = []
    if belief_state['processes']['access_grant'].get("confidence",0.0) > THRES:

        if belief_state['methods']['finished'].get("confidence",0.0) < THRES:
            flag = 1
            # If approval is not present
            if belief_state['usr_slots']['approval'].get("confidence",0.0) < THRES:
                flag=0
                info_fields = ["seniority","team_name",'id']


            # If request_number is not present
            if belief_state['usr_slots']['request_number'].get("confidence",0.0) < THRES:
                flag = 0
                info_fields = ["supervisor","supervisor_email",'team_name']

    info_values = [context_info.get(f) for f in info_fields]
    fields = info_fields[:3]
    for f in context_info:
        if len(fields)>=3:
            break
        if f in info_fields:
            continue
        info_values.append(context_info.get(f))
        fields.append(f)

    for i,f in enumerate(fields):
        if f=='team_name':
            fields[i] = 'department_name'
    return fields, info_values



def get_feedback_image_links():

    img_link = configdata['static_server']+'/static/images/'
    #return [img_link+'great.png',img_link+'bad.png']
    return ['https://upload.wikimedia.org/wikipedia/commons/thumb/7/79/Face-smile.svg/48px-Face-smile.svg.png',
            'https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Face-plain.svg/48px-Face-plain.svg.png']
