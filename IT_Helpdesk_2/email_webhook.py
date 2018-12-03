import sys
import smtplib
import imaplib
import email
import requests
import re
import numpy as np 
from time import sleep
import logging
import json
import base64

from googleapiclient.discovery import build
from httplib2 import Http
from oauth2client import file, client, tools

from requests.exceptions import ConnectionError

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

class EmailWebhook(object):
    def __init(self,*args,**kwargs):
        pass

    def send_message(self,msg,toaddrs):
        raise NotImplementedError("Send message is not implemented")

    def get_unseen_messages(self):
        raise NotImplementedError("Get unseen messages is not implemented")

    def send_message(self,toaddrs,subject,body):
        raise NotImplementedError("send message is not implemented")

    def label_email(self,email_id,remove=None,add=None):
        pass

class GmailWebhook(EmailWebhook):
    def __init__(self,token_file,*args,**kwargs):

        super().__init__(*args,**kwargs)
        #SMTP
        self.fromaddr = 'customerservice@suprath.com'
        self.token_file = token_file

        self.service = self.get_service(self.token_file)


    @staticmethod
    def get_service(token_file):
        store = file.Storage(token_file)
        creds = store.get()
        service = build('gmail', 'v1', http=creds.authorize(Http()))
        return service
    
    def sendmail(self,msg,threadId):
        if threadId is not None:
            body = {"raw":msg.decode(),"threadId":threadId}
        else:
            body = {"raw":msg.decode()}
        self.service.users().messages().send(userId='me',body=body).execute()

    def send_message(self,toaddrs,subject,body,CC=None,threadId=None):
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = self.fromaddr
        msg['To'] = toaddrs
        if CC is not None and CC != '':
            msg['Cc'] = ','.join(CC)

        msg.attach(MIMEText(body, 'html'))

        msg_encoded = base64.urlsafe_b64encode(msg.as_string().encode('ascii'))


        self.sendmail(msg_encoded,threadId)

    @staticmethod
    def get_first_text_block(email_message_instance):
        maintype = email_message_instance.get_content_maintype()
        if maintype == 'multipart':
            for part in email_message_instance.get_payload():
                if part.get_content_maintype() == 'text':
                    return part.get_payload()
        elif maintype == 'text':
            return email_message_instance.get_payload()

    def get_clean_text(self,email):
        text = self.get_first_text_block(email)
        text = re.sub(r'\r',r'',text)
        return text 

    def get_email(self,email_id):
        raw =  self.service.users().messages().get(id=email_id,userId='me',format='raw').execute()['raw']
        raw_decoded = base64.urlsafe_b64decode(raw)  
        email_message = email.message_from_bytes(raw_decoded)
        return email_message

    def label_email(self,email_id,remove=None,add=None):
        remove = [] if remove is None else remove
        add = [] if add is None else add
        self.service.users().messages().modify(userId='me',id=email_id,
            body={'removeLabelIds': remove, 'addLabelIds':add}).execute()

    def get_unseen_messages(self):
        emails = []
        result = self.service.users().messages().list(userId='me',labelIds=['UNREAD']).execute()

        if 'messages' not in result:
            return []

        for message in result['messages']:
            email_id = message['id']
            thread_id = message['threadId']

            email_message = self.get_email(email_id)
            
            toaddrs = email.utils.parseaddr(email_message['From'])[-1]
            try:
                CC = [email.utils.parseaddr(add)[-1] for add in email_message['Cc'].split(',')]
            except AttributeError:
                CC = ['']
            emails.append({'sender':toaddrs,
                            'CC': CC,
                            'message':self.get_clean_text(email_message),'subject':email_message['subject'],
                            'threadId':thread_id,
                            'emailId': email_id})

        return emails

        

class idare_interface(object):

    def __init__(self,idare_ip,idare_port,email_client):
        self.idare_ip = idare_ip
        self.idare_port = idare_port
        self.idare_api = 'http://'+idare_ip+':'+idare_port+'/email/'
        self.email_client = email_client

    def listen_and_send(self):
        logger.info('Started listening to new messages')
        while True:
            logging.disable(logging.INFO) # Otherwise google api prints result everytime which becomes high volume
            emails = self.email_client.get_unseen_messages()
            logging.disable(logging.NOTSET) # Remove the logging restriction 
            for Email in emails:
                try:
                    logger.info(f'NEW_EMAIL:\n{Email}\n')
                    #following is a temporary line
                    sender_id = Email['threadId']
                    data = {'q':Email['message'],
                            'email_id': Email['sender']}
                    try:
                        print(sender_id,data)
                        r=requests.get(self.idare_api+sender_id+'/respond',data)

                        if r.status_code == requests.codes.ok:
                            body = r.json()['text']

                            logger.info(f'IDARE_REPLY:\n{body}\n')

                            self.email_client.send_message(Email['sender'],'RE: '+Email['subject'],
                                                body,CC=Email['CC'],
                                                   threadId = Email['threadId'] )

                            self.email_client.label_email(Email['emailId'],remove=['UNREAD'])
                        else:
                            logger.error(f"Got {r.status_code} status code from iDARE: {r.content}") 
                            raise Exception("iDARE responded problematically")
                    except ConnectionError as e:
                        self.email_client.label_email(Email['emailId'],add=['UNREAD'])
                        logger.error('Please ensure iDARE server is up, can not connect. Error: {}'.format(e))
                        raise Exception('iDARE server not up')
                except Exception as e:
                    self.email_client.label_email(Email['emailId'],add=['UNREAD'])
                    logger.error(e.__class__.__name__+str(e))
                    logger.warning('Sleeping for 10 seconds')
                    sleep(10)
        sleep(1)

if __name__ == '__main__':
    args = sys.argv
    if len(sys.argv) > 1:
        env_config_path = sys.argv[1]
    else:
        raise Exception("Please give environment configuration json as argument")

    with open(env_config_path) as f:
        env_config = json.load(f)

    idare_ip = env_config['server_ip']
    idare_port = env_config['server_port']

    email_client = GmailWebhook(env_config['token_file'])


    logging.basicConfig(level='DEBUG')
    idare_interface(idare_ip,idare_port,email_client).listen_and_send()
