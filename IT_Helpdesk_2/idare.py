from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json
import logging
import os
import warnings
import typing
import numpy as np
import time
import regex as re

from typing import Any, List, Dict, Text, Optional, Tuple
from collections import defaultdict

from rasa_core import utils
from rasa_core.policies.idare_policy import IDARE
from rasa_core.featurizers import TrackerFeaturizer
from rasa_core.events import UserUttered, ActionExecuted, BotUttered
from rasa_core.exceptions import APIError
from smart_models import get_sentiment,smart_routing,information_retrieval,decision_making,redirect_url
from smart_models import get_feedback_image_links

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import keras
    from rasa_core.domain import Domain
    from rasa_core.trackers import DialogueStateTracker

def email_preprocessor(email,email_id=None):

    clauses = re.split(r' and | but | or |,|\.|\?|\n', email)

    strippedClauses = []
    if email_id is not None:
        strippedClauses.append(email_id.lstrip().strip())

    for eachClause in clauses:
        if(eachClause != ""):
            strippedClauses.append(eachClause.lstrip())
    return strippedClauses

def get_greeting(name):

    def to_camelcase(string):
        return ' '.join([s[0].title()+s[1:] for s in string.split()])

    if name is None:
        return "Hi,\n\n"

    return f"Dear {to_camelcase(name)},\n\n"

    return f"Hi {name},\n\n"


def to_html(body):
    html = """<html>
      <head></head>
      <body style="text-align:justify;">
       """+body+"""
      </body>
    </html>
    """
    return html


def get_footer(re_url):
    images = get_feedback_image_links()
    body = """
        <br>
        <span style="font-family: serif;"><i>Were you satisfied with this resolution?</i></span>
        <br>
        <div id="up" style="display:inline-flex;">
            <a href="http://google.com">
                <img src = \""""+images[0]+"""\" alt = "resolved" style="height:3em;width:3em" >
            </a>
        </div>
        <div id="down" style="display:inline-flex;">
            <a href="http://google.com">
                <img src = \""""+images[1]+"""\" alt = "unresolved" style="height:3em;width:3em" >
            </a>
        </div>
        <br>
        <p style="font-family: serif; font-size: x-small; text-align: justify;">
            <i>This is an AI generated email. In case the problem remains unresolved our
             customer care representative will get in touch with you within 24 hours. Please note down the
            reference number 612123151 for future reference.</i><br><br>
        </p>
        <p>
            Regards,
            <br>
            Customer Care<br>
            Suprath Technologies
        </p>"""

    if re_url is not None and re_url is not "":
        body = """<br>
        <a href='{}'>Click here to chat directly with me, your virtual agent, for faster resolution</a><br>""".format(re_url)\
                +body
    return body

class KerasIDarePolicy(IDARE):    

    def __init__(self,*args,**kwargs):

        self.information_retrieval = information_retrieval
        super().__init__(*args,**kwargs)

    def predict_next_email(self, tracker, domain):
        """Returns UserUtterances with updated belief state, and the predicted
        action events"""
        # type: (DialogueStateTracker, Domain) -> List[float]

        belief_state = tracker.belief_state

        # Now get the message
        actions, links, confidences, context_info_field, context_info_value, insights = self.predict_messages(belief_state,tracker)

        tracker.ha_id = None
        tracker.mode = 'virtual_agent'


        if actions is not None:
            usr_text = "<br>".join([re.sub(r'\n','<br>',self.NLG_top3(*iterated)[0]) if iterated[0]!='greet' \
                                    else "How may I help you?" for iterated in actions])

        else:
            usr_text = self.NLG_top3('sorry',['repeat'])[0]+"<br>"+\
                        "This incidence will be reported to supervisor in order to serve you better."

        usr_text = '<p>'+get_greeting(belief_state['usr_slots']['name'].get('value'))+'</p><p>'+usr_text+'</p>'
        re_url =  redirect_url(tracker)

        footer = get_footer(re_url)


        usr_message = defaultdict(list)
        usr_message['text'] = to_html(usr_text+footer)


        if links[0]:
            [usr_message["image"].append(l) \
            for l in links[0] if l!="" and any([l.endswith(x)\
                             for x in [".png",".jpeg",".jpg"]])]

            [usr_message["ui_actions"].append(l) \
            for l in links[0] if l!="" and not any([l.endswith(x)\
                             for x in [".png",".jpeg",".jpg"]])]
        usr_message["event_id"] = 'action_{}'.format(time.time())

        confidence = 1.0
        return (usr_message,None), confidence

         
    def predict_next_message(self, tracker, domain):
        """Returns UserUtterances with updated belief state, and the predicted
        action events"""
        # type: (DialogueStateTracker, Domain) -> List[float]

        belief_state = tracker.belief_state
        # Now get the message
        actions, links, confidences, context_info_field, context_info_value, insights = self.predict_messages(belief_state,tracker)
        _ha_message = self.form_assist_messages(actions, links, \
                                                confidences, tracker.ha_id, tracker, \
                                                context_info_field, context_info_value,insights)

        if tracker.latest_sender == 'user':
            latest_intent = tracker.latest_message.nlu_act
        else:
            latest_intent = tracker.latest_message.action_act
        usr_actions, ha_id, mode = smart_routing(self,_ha_message,actions,\
                                latest_intent,confidences,tracker.ha_id, tracker.mode,tracker)
        tracker.ha_id = ha_id
        tracker.mode = mode

        usr_message = None

        if usr_actions is not None:
            usr_message = defaultdict(list)
            usr_text = self.NLG_top3(*usr_actions[0])[0]
            usr_message['text'] = usr_text
            if mode == 'virtual_agent':
                if links[0]:
                    [usr_message["image"].append(l) \
                    for l in links[0] if l!="" and any([l.endswith(x)\
                                     for x in [".png",".jpeg",".jpg"]])]

                    [usr_message["ui_actions"].append(l) \
                    for l in links[0] if l!="" and not any([l.endswith(x)\
                                     for x in [".png",".jpeg",".jpg"]])]
                usr_message["event_id"] = 'action_{}'.format(time.time())


        ha_message = None

        if mode == 'agent_assist':
            ha_message = _ha_message

        #we also need to store the mode and ha_id changes in the events
        if ha_message:
            ha_message["mode"] = mode
            ha_message["ha_id"] = ha_id

        confidence = 1.0
        return (usr_message,ha_message), confidence
    

    @staticmethod
    def get_current_topic(tracker):
        max_process_conf = 0.0
        result = ''
        for name,process in tracker.belief_state['processes'].items():
            if process['confidence']>max_process_conf:
                max_process_conf =process['confidence']
                result = name
        max_process_conf = "{:0.2f}".format(max_process_conf)
        return result,max_process_conf

    @staticmethod
    def get_ticket_details(tracker):
        """Populating ID, date (of chat open time), category, sub_category, and issue (details)"""
        thres = 0.7
        result = {}
        # For ID
        if tracker.belief_state['usr_slots']['ID'].get('confidence') > thres:
            result['ID'] = tracker.belief_state['usr_slots']['ID'].get('value')
        # For name
        if tracker.belief_state['usr_slots']['name'].get('confidence') > thres:
            result['name'] = tracker.belief_state['usr_slots']['name'].get('value')
        # For name
        if tracker.belief_state['usr_slots']['email_id'].get('confidence') > thres:
            result['email_id'] = tracker.belief_state['usr_slots']['email_id'].get('value')
        # For request open time
        import time
        chat_open_time = time.ctime(tracker.events[0].timestamp)
        result['date'] = chat_open_time
        # For category
        topic,_ = KerasIDarePolicy.get_current_topic(tracker)
        if topic != '':
            result['category'] = topic

        return result


    def form_assist_messages(self,actions, links, confidences, ha_id, tracker,
                                context_info_field ,context_info_value,insights):
        ha_message = {}
        if tracker.latest_sender == 'user':
            ha_message.update({ 'text': tracker.latest_message.text})
            slot_name, slot_value = [],[]
            intents = [tracker.latest_message.nlu_act['name']]

            other_conf = 1
            for slot in tracker.latest_message.nlu_slots:
                if slot['confidence']>0.5:
                    if slot['value'] is None or slot['value'] == 'None':
                        intents.append(slot['slot'])
                        other_conf *= slot['confidence']
                    else:
                        slot_name.append(slot['slot'])
                        slot_value.append(slot['value'])


            ha_message.update({'intent': "-".join(intents),\
                              'slots': [slot_name,slot_value],
                            'confidence': \
                            '{:0.2f}'.format(other_conf*tracker.latest_message.nlu_act['confidence'])})

        if actions == None or actions == []:
            ha_message.update({ 'recom':[""]*3})
            insights_ = ["As per resolution procedure"]*3
        else:
            random_state = np.random.permutation(3)
            selected_messages = np.random.permutation(len(actions))[:3]
            filled = [0]*len(actions)
            done = [False]*len(actions)
            insights_ = []
            recom = [self.NLG_top3(*actions[i],random_state=random_state)[filled[i]] for i in selected_messages]
            insights_ = [insights[i][0] for i in selected_messages]
            for i in selected_messages:
                filled[i]+=1
            while True:
                if len(recom)>=3 or False not in done:
                    break
                for i in range(len(actions)):
                    if done[i]:
                        continue
                    recos = self.NLG_top3(*actions[i],random_state=random_state)
                    if filled[i] == len(recos):
                        done[i] = True
                        continue
                    recom.append(recos[filled[i]])
                    filled[i]+=1
                    if len(recom)==3:
                        break
            if len(recom) < 3:
                logger.error(("Couldnt fill 3 recommendations, check NLG_top3. Filling other recommendations with "
                                            "sorry I dont understand"))
                recom[len(recom):3] = ["Sorry I dont understand what you mean."]*3

            ha_message.update({
                              "recom": recom
                             })

        #Follwing is a temporary code
        three_links = [_l for l in links for _l in l if _l!='' and 'UI_ACTION' not in _l][:3]
        while len(three_links)<3:
            three_links.append('')

        ha_message.update({"links": three_links})
        ha_message.update({"context_field" : context_info_field})
        ha_message.update({"context_info" : [str(val) for val in context_info_value]})
        ha_message.update({"insights" : insights_})
        #Some housekeeping keys
        ha_message["sender_id"] = tracker.sender_id
        ha_message["chat_open_time"] = tracker.events[0].timestamp
        ha_message["topic"],ha_message["topic_confidence"] = self.get_current_topic(tracker)
        ha_message["event_id"] = 'assist_{}'.format(time.time())

        # Ticket population
        ha_message["ticket_details"] = self.get_ticket_details(tracker)
        return ha_message

    @staticmethod
    def get_slot_from_nlu_slots(nlu_slots, slot_name):
        for l in nlu_slots:
            if l.get('slot') == slot_name:
                return l

    def NLG_top3(self,act,slots,random_state = None):
        if random_state is None:
            random_state = np.array([0,1,2])
        result = None
        if len(slots) == 2:
            if act == 'inform' and "password_reset" in slots and 'finished' in slots:
                result = ["Password reset has been completed , you will receive your new password through mail.",
                        "I have reset your password, you can new set password now.",
                        "The password reset is done, you'll receive a mail for setting up new password."]


            if act == 'inform' and "access_grant" in slots and 'finished' in slots:
                result = ["I have given access to the machine.",
                        "Access to the machine is now granted.",
                        "Machine access is now granted."]


            if act == 'request' and 'pdf_converter' in slots and 'login' in slots:
                result = ["I can see that you have a valid license for Adobe Acrobat Professional. Can you please login to that?",
                        "You have a valid license for Adobe Acrobat Professional. Can you please login?",
                        "You have a valide license for Adobe. Could you please login?"]

            if act == 'confirm' and 'login' in slots and 'wait' in slots:
                result = ["Please tell me if you are able to login or not.",
                        "Please confirm if you can login.",
                        "Please tell me once you are able to login"]


        if len(slots) == 1:


            if act == 'request' and 'pdf_converter' in slots:
                result = ["Which PDF converter are you using?",
                        "What PDF converter is being used by you?",
                        "Can you tell me which PDF converter you are using?"]

            if act == 'confirm' and 'resolved' in slots:
                result = ["Are you able to convert excel files to PDF now?",
                        "Can you convert excel to PDf now?",
                        "Check if you can convert excel to pdf now."]

            if act == 'instruction' and 'PLIL1' in slots:
                result = ["Please follow the instructions in the image to login.\n\n"+
                        "You can give the username and password you received with the license email."]*3
                        
            if act == 'instruction' and 'xls2pdf_harddrive' in slots:
                result = ["Please ensure that you save the PDF directly to your hard drive and not to a shared drive."]*3

            if act == 'request' and "approval" in slots:
                result = ["Do you have the approval from your supervisor?",
                        "Can you please tell me if your supervisor has approved the access grant?",
                        "Have you received approval from your supervisor?"]

            if act == 'request' and "request_number" in slots:
                result = ["Can i have the request number please?",
                        "Do you have the request number?",
                        "Could you please tell me the request number?"]

            if act == 'request' and "ID" in slots:
                result = ["Can I have your user id?",
                        "Please tell me your user ID.",
                        "What is the user ID associated with the account?"]


            if act == 'request' and 'send_otp' in slots:
                result = [("An email will be sent to you with a security code. "
                         "I will need the code to reset the password. "
                         "Can I go ahead with it?"),

                        ("I will have system send you a verification email. "
                         "I will need the security code in that email. "
                         "Can I go ahead?"),

                        ("An email will be sent to you for security code. "
                         "The code is needed for verification. "
                         "Can I go ahead and send it?")]

            if act == 'request' and 'security_code' in slots:
                result = ["The email has been sent, can you tell me the security code?",
                        "Please tell me the security code that you just received.",
                        "Can you tell me the security code you received in the email?"]

            if act == 'required' and "approval" in slots:
                result = ["Approval from supervisor is needed for granting access.",
                        "You need approval from supervisor for accessing the machine.",
                        "Without supervisor approval I can't proceed further."]


            if act == 'required' and "request_number" in slots:
                result = ["You'll have to obtain request number from your supervisor.",
                        "Request number is needed to proceed further.",
                        "Please ask supervisor for request number for access grant."]


            if act == 'required' and 'ID' in slots:
                result = ["User ID is required before we proceed further.",
                        "I can't go forward without knowing your user ID.",
                        "I have to know your user ID before I can proceed further."]

            if act == 'required' and 'send_otp' in slots:
                result = ["Without the security code from email, I wont be able to go forward.",
                        "I need to know the security code, otherwise I cant process your request.",
                        "Without the security code from email, I won't be able to go ahead."]

            if act == 'confirm' and 'login_able' in slots:
                result = ["Can you check if you are able to login.",
                        "Are you able to login?",
                        "Please check if you can login."]
            if act == 'deny' and 'admin_access' in slots:
                result = ["No as per policy we don't give admin rights to the users on this machine.",
                        "Sorry, admin rights are not available as per the policy.",
                        "I am afraid you dont have administrator access according to the policy."]
            
            if act == 'gratitude' and 'feedback' in slots:
                result = ["Thanks for the feedback.",
                        "Thank you. Your feedback is highly appreciated.",
                        "Thank you for your valuable feedback."]

            if act == 'sorry' and 'repeat' in slots:
                result = ["I am sorry, I don't understand what you mean.\nCan you please repeat?",
                        "I apologize but I don't understand what you mean.\nPlease try again.",
                        "I am afraid I don't know what you mean, care to try again?"]

        if len(slots) == 0:

            if act == 'greet' and slots == []:
                result = ["Hi, how may I help you?",
                        "Hey there, how may I assist you?",
                        "Welcome to IT helpdesk"]

            if act == 'activation_intiated' and slots == []:
                result = ["I have initiated the activation request, please check within one day.",
                        "The activation request is initiated, please check soon",
                        "I have started the activation process"]

            if act == 'sorry' and slots == []:
                result = ["I am sorry.",
                        "I apologize.",
                        "My mistake."]
            if act == 'bye' and slots == []:
                result = ["Bye.",
                        "Happy to help, bye!",
                        "I am here to help. Bye!"]

            if act == 'anythingelse' and slots == []:
                result = ["Is there anything else?",
                        "Can I help you with anything else",
                        "Do you need any other help?"]

            if act == 'feedback' and slots == []:
                result = ["Please rate this conversation.",
                        "Please give us your valuable feedback.",
                        "How would you rate this conversation?"]

            if act == 'redirect' and slots == []:
                result = [("I am sorry, I am unable to help you with this request. \n\n"
			              "I will escalate this issue to an Engineer. Please wait")]*3


        if result is None:
            logger.error("{} and {} not present in nlg".format(act,slots))
            return ["I am sorry I dont understand what you mean"]*3
        return [result[i] for i in random_state]

    def predict_messages(self,belief_state,tracker):

        THRES = 0.6
        results = []
        links = []
        context_info_field = [""]
        context_info_value = [""]
        insight = []
        context_info_field, context_info_value = decision_making(belief_state)
        # Lower layers
        #If method is bye, say bye
        if belief_state['methods']['bye'].get("confidence",0.0) > THRES:
            if belief_state['sys_goals']['feedback'].get("confidence",0.0) < THRES:
                results.append(('feedback',[]))
                insight.append([""]*3)
                links.append(["<UI_ACTION_FEEDBACK>"])
            else:
                results.append(('bye',[]))
                insight.append([""]*3)
                links.append([""]*3)
        # If method is none, say how may I help you?
        if belief_state['methods']['none'].get("confidence",0.0) > THRES:
            results.append(('greet',[]))
            insight.append([""]*3)
            links.append([""]*3)

        # If method is finished, ask is there anything else?
        if belief_state['methods']['finished'].get("confidence",0.0) > THRES:
            if belief_state['processes']['password_reset'].get("confidence",0.0) > THRES:
                results.append(('anythingelse',[]))
                insight.append([""]*3)
                links.append([""]*3)
            elif belief_state['processes']['access_grant'].get("confidence",0.0) > THRES:
                if belief_state['usr_slots']['login_able'].get('confidence',0.0) > THRES:
                    tok = belief_state['usr_slots']['approval'].get("value","")
                    if tok == self.tokens_inverse.get('unavailable'):
                        results.append(('sorry',[]))
                        insight.append([""]*3)
                        links.append([""]*3)
                    elif belief_state['usr_slots']['admin_access'].get('confidence',0.0) > THRES:
                        results.append(('deny',['admin_access']))
                        insight.append([""]*3)
                        links.append(self.information_retrieval(0))
                    else:
                        results.append(('anythingelse',[]))
                        insight.append([""]*3)
                        links.append([""]*3)
                elif belief_state['usr_slots']['admin_access'].get('confidence',0.0) > THRES:
                    results.append(('deny',['admin_access']))
                    insight.append([""]*3)
                    links.append(self.information_retrieval(0))
                else:
                    results.append(('confirm',['login_able']))
                    insight.append(["Supervisor approval is required for access grant"]*3)
                    links.append([""]*3)
            elif belief_state['processes']['Software Troubleshooting'].get("confidence",0.0) > THRES:
                results.append(('anythingelse',[]))
                insight.append([""]*3)
                links.append([""]*3)
        # If process is access grant and if it isn't finished
        if belief_state['processes']['access_grant'].get("confidence",0.0) > THRES:

            if belief_state['methods']['finished'].get("confidence",0.0) < THRES:
                flag = 1
                # If approval is not present
                if belief_state['usr_slots']['approval'].get("confidence",0.0) < THRES:
                    flag = 0
                    results.append(('request',['approval']))
                    insight.append(["Supervisor approval is required for access grant"]*3)
                    links.append(self.information_retrieval(1))
                else:
                    tok = belief_state['usr_slots']['approval'].get("value","")
                    if tok == self.tokens_inverse.get('unavailable'):
                        flag = 0
                        results.append(('required',['approval']))
                        insight.append(["Supervisor approval is required for access grant"]*3)
                        links  = [self.information_retrieval(1)]

                # If request_number is not present
                if belief_state['usr_slots']['request_number'].get("confidence",0.0) < THRES:
                    flag = 0
                    insight.append(["Request number should be obtained from supervisor"]*3)
                    results.append(('request',['request_number']))
                    links.append(self.information_retrieval(1))
                else:
                    tok = belief_state['usr_slots']['request_number'].get("value","")
                    if tok == self.tokens_inverse.get('unavailable'):
                        flag = 0
                        insight.append(["Request number should be obtained from supervisor"]*3)
                        results = [('required',['request_number'])]
                        links  = [self.information_retrieval(1)]

                if flag == 1: 
                    results.append(('inform',['access_grant','finished']))
                    links.append(self.information_retrieval(1))
                    insight.append(["Access should now be granted to the employee"]*3)


        # If process is Software Troubleshooting and if it isn't finished
        if belief_state['processes']['Software Troubleshooting'].get("confidence",0.0) > THRES:

            if belief_state['methods']['finished'].get("confidence",0.0) < THRES:
                flag = 1
                # If the problem is issue-convert-xls2pdf
                if belief_state['usr_slots']['issue'].get("confidence",0.0) > THRES and \
                    belief_state['usr_slots']['convert'].get("confidence",0.0) > THRES and \
                    belief_state['usr_slots']['xls2pdf'].get("confidence",0.0) > THRES:
                    # If pdf_converter is not known
                    if belief_state['usr_slots']['pdf_converter'].get("confidence",0.0) < THRES:
                        flag=0
                        results.append(('request',['pdf_converter']))
                        insight.append(["It is necessary to know which pdf converter is being used"]*3)
                        links.append(self.information_retrieval(2))

                    # If user is not logged in
                    if belief_state['usr_slots']['login'].get("confidence",0.0) < THRES:
                        flag = 0
                        results.append(('request',['pdf_converter','login']))
                        insight.append(["It is necessary to login in the software for conversion"]*3)
                        links.append(self.information_retrieval(2))
                    else:
                        #If user has asked how to login
                        if belief_state['usr_slots']['howto'].get("confidence",0.0) > THRES:
                            #If user has told to wait
                            if belief_state['usr_slots']['wait'].get("confidence",0.0) > THRES:
                                flag = 0
                                results.append(('confirm',['login','wait']))
                                insight.append(["Wait while user tries to login"]*3)
                                links.append(self.information_retrieval(2))
                            else:
                                flag = 0
                                results.append(('instruction',['PLIL1']))
                                insight.append(["These are instruction for logging in"]*3)
                                links  = [self.information_retrieval(3)]

                    if belief_state['sys_goals']['resolved'].get("confidence",0.0) > THRES:
                        flag = 0
                        results.append(('confirm',['resolved']))
                        insight.append(["Ask user again if they are able to convert or not"]*3)
                        links.append(self.information_retrieval(2))


                    # If not xls2pdf_harddrive is not confirmend
                    if belief_state['usr_slots']['xls2pdf_harddrive'].get("confidence",0.0) < THRES:
                        flag = 0
                        insight.append(["PDF shouldn't be saved to the shareddrive"]*3)
                        results.append(('instruction',['xls2pdf_harddrive']))
                        links.append(self.information_retrieval(2))

                    if flag==1:
                        flag = 0
                        insight.append(["At this point we dont know how to solve"]*3)
                        results.append(('redirect',[]))
                        links.append(self.information_retrieval(2))



        # If process is account password_reset and if it isn't finished
        if belief_state['processes']['password_reset'].get("confidence",0.0) > THRES and \
                       belief_state['methods']['finished'].get("confidence",0.0) < THRES:
            flag = 1
            # If ID is not known 

            if belief_state['usr_slots']['ID'].get("confidence",0.0) < THRES:
                if belief_state['usr_slots']['unavailable_slots'].get("confidence",0.0) > THRES \
                    and belief_state['usr_slots']['unavailable_slots'].get("value",None) == 'id':
                    flag = 0
                    results = [('required',['ID'])]
                    insight.append([""]*3)
                    links  = [self.information_retrieval(1)]
                else:
                    flag = 0
                    results.append(('request',['ID']))
                    insight.append([""]*3)
                    links.append(self.information_retrieval(1))         

            # If send_otp is not known
            if belief_state['usr_slots']['send_otp'].get("confidence",0.0) < THRES:
                if belief_state['usr_slots']['unavailable_slots'].get("confidence",0.0) > THRES \
                    and belief_state['usr_slots']['unavailable_slots'].get("value",None) == 'send_otp':
                    flag = 0
                    results = [('required',['send_otp'])]
                    insight.append([""]*3)
                    links  = [self.information_retrieval(1)]
                else:
                    flag = 0
                    results.append(('request',['send_otp']))
                    insight.append([""]*3)
                    links.append(self.information_retrieval(1))
            # If security code is not present
            if belief_state['usr_slots']['security_code'].get("confidence",0.0) < THRES:
                if belief_state['usr_slots']['unavailable_slots'].get("confidence",0.0) > THRES \
                    and belief_state['usr_slots']['unavailable_slots'].get("value",None) == 'security_code':
                    flag = 0
                    results = [('required',['security_code'])]
                    insight.append([""]*3)
                    links  = [self.information_retrieval(1)]
                else:
                    flag = 0
                    results.append(('request',['security_code']))
                    insight.append([""]*3)
                    links.append(self.information_retrieval(1))

            if flag == 1: 
                results.append(('inform',['password_reset','finished']))
                links.append(self.information_retrieval(1))
                insight.append([""]*3)


        if results == []:
            return None,[""]*3,1.0,[""],[""],[[""]*3]
        return results, links, [1.0]*len(results), context_info_field, context_info_value, insight

