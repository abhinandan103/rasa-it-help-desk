import argparse
import sys
import requests
import _pickle as pickler
import json
import os
import logging

import redis
import zipfile
import pymysql
from gevent.pywsgi import WSGIServer
from flask import Flask, request, jsonify
from requests.exceptions import ReadTimeout,ConnectionError
from simplejson.errors import JSONDecodeError

from suprath_nlu.wrapper import Interpreter
from suprath_nlu import Interpreter as Sinterpreter

logger = logging.getLogger(__name__)


env_config=None

def get_feedback_info():
    connection = getDBConnection()
    try:
        with connection.cursor() as cursor:
            sql=("SELECT * from feedback.feedback WHERE trained_status = '0'")
            cursor.execute(sql)
            result = cursor.fetchall()  
    finally:
        connection.close()

    return result


def set_train_status(raw_data):

    if len(raw_data) == 0:
        return


    connection = getDBConnection()
    try:
        with connection.cursor() as cursor:
            sql=("UPDATE feedback.feedback SET trained_status = '1'"
                 f" WHERE id in ({','.join([str(r['id']) for r in raw_data])})")
            print(sql)
            cursor.execute(sql)

            connection.commit()
            logger.debug("Updated trained feedback successfully")
    except pymysql.IntegrityError as e:
        logger.warn("Failed to update trained feedback in pymysql {}".format(e))
    finally:
        connection.close()


def get_nlu_train_data(raw_data,redis_ip,redis_port):
    processed_data=[]
    redis_add = 'http://'+redis_ip+':'+redis_port+'/feedback'
    for d in raw_data:
        try:
            if d['feedback_type'] == 'nlu_feedback':
                try:
                    r = requests.get(redis_add,params={"sender_id":d['conv_id'],
                                                    "event_id":d['event_id'],
                                                    "feedback_type":'nlu'})
                except ConnectionError as e:
                    logger.error(str(e))
                    continue
                data = r.json()
                try:
                    processed_data.append([data['intent'].split('-')[0].strip(),
                                             data["text"] , 
                                             int(data["reward"])])
                except KeyError as e:
                    logger.error(str(e))
                    continue
        except ValueError as e:
            logger.error(str(e))
            continue
    print("Got the following data {}".format(processed_data))
    return processed_data



def create_app(port,server_ip,server_port,redis_ip,redis_port,nlu_train_file,path_to_glove):

    app = Flask(__name__)
    server_url = "http://"+server_ip+':'+server_port
    model_directory = "./nlu_model/default/current"
    feedback_file_loc = model_directory+'/data/feedback_data.json'
    logger.debug('Loading Suprath nlu')
    nlu_model = Sinterpreter(path_to_glove = path_to_glove)
    try:
        nlu_model.load(path_to_dir = model_directory)
    except OSError:
        logger.warning('Model doesnt already exist, please train suprath nlu model once')

    @app.after_request
    def after_request(response):
      response.headers.add('Access-Control-Allow-Origin', '*')
      response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
      response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
      return response

    @app.route("/run")
    def start():
        """Runs the server"""
        logger.info("Runnning the server")
        try:
            r = requests.post(server_url+'/run',timeout=5)
        except ReadTimeout:
            logger.info("Server now running")
            return jsonify({"success":1})
        except Exception as e:
            logger.error(str(e))
            return "Exception %s"%e
        logger.error("Some error occurred")
        return "Error in starting the server"

    @app.route("/ssg",methods=['GET', 'OPTIONS'])
    def ssg():
        logger.info("Starting ssg training")
        args = request.args
        if not args:
            return "Invalid request"
        ssg_file = args.get('file')
        if not ssg_file:
            return "Invalid request"

        with open(nlu_train_file,'r') as f:
            nlu_data = json.load(f)

        try:
            with open(ssg_file,'r') as f:
                ssg_data = json.load(f)
        except Exception:
            with open(ssg_file,'r') as f:
                ssg_data = eval(f.read())
        except Exception as e:
            logger.error(str(e))
            return str(e)

        nlu_data['rasa_nlu_data']['common_examples']+=(ssg_data)

        with open(nlu_train_file,'w') as f:
            json.dump(nlu_data,f,indent=4, sort_keys=True)
        
        logger.info("Success SSG training")
        return jsonify({"success":1})

    @app.route("/train_nlu",methods=['GET', 'OPTIONS'])
    def process_and_train_nlu():
        """Processes all the nlu feedback and train the nlu model"""
        logger.info("Training nlu")
        nlu_model_config_file = "nlu_model_config.yml"

        Interpreter.train(nlu_train_file,'./nlu_model',nlu_model_config_file, path_to_glove)

        #Finally setting all the trained data train_status to 1
        logger.info("Training nlu successfull")
        return jsonify({"success":1})

    @app.route('/feedback_train',methods=['GET','OPTIONS'])
    def feedback_train():
        logger.info("Starting feedback training")
        raw_data = get_feedback_info()
        #if not raw_data or len(raw_data) < 1:
        if not raw_data or len(raw_data) < 1:
            return jsonify({"Error":"No data available to train"})
        if not os.path.exists(os.path.dirname(feedback_file_loc)):
            try:
                os.makedirs(os.path.dirname(feedback_file_loc))
            except OSError as exc: # Guard against race condition
                if exc.errno != os.errno.EEXIST:
                    raise exc

        train_data = get_nlu_train_data(raw_data,redis_ip,redis_port)
        update_nlu_training(train_data,feedback_file_loc)

        nlu_model.train_feedback()

        #Finally setting all the trained data train_status to 1
        set_train_status(raw_data)
       
        logger.info("Feedback training successfull")
        return jsonify({'success':1})

    @app.route("/deploy_nlu",methods=['GET', 'OPTIONS'])
    def deploy_nlu():
        """ Deploys the nlu model to the server"""
        #First compressing
        logger.info("Depoying NLU..")
        zout = zipfile.ZipFile('nlu.zip', "w", zipfile.ZIP_DEFLATED) 
        nlu_dir = model_directory+'/'
        for fname in os.listdir(nlu_dir):
            print ("writing: ", fname)
            zout.write(nlu_dir+fname,arcname=os.path.join('./', fname))
        zout.close()

        file = {'nlu_model': open('nlu.zip','rb')}
        url = 'http'
        try:
            result = requests.post(server_url+'/load_nlu', files=file)
            logger.info("Deployed NLU successfully")
            return jsonify(result.json())
        except JSONDecodeError:
            string = "Got this from server: {}".format(result.status_code)
            logger.info(string)
            return string
        except ConnectionError as e:
            logger.error(e.__class__.__name__+str(e))
            return str(e)

    def update_nlu_training(train_data,feedback_file_loc):
        feedback = []
        
        for dat in train_data:
            feedback.append({'act': dat[0], 'feedback': int(dat[2]), 'query': dat[1]})
        with open(feedback_file_loc,'w') as f:
            json.dump(feedback,f)

    return app

def getDBConnection():
    connection = pymysql.connect(host=env_config['db_host'],
                                user=env_config['db_username'],
                                password=env_config['db_password'],
                                db='agentdb',
                                charset='utf8mb4',
                                cursorclass=pymysql.cursors.DictCursor)
    return connection

if __name__=='__main__':
    if len(sys.argv) > 1:
        env_config_path = sys.argv[1]

    with open(env_config_path) as f:
        env_config = json.load(f)

    port = int(env_config['admin_port'])

    if env_config.get('nlu_training_file'):
        nlu_file = env_config.get('nlu_training_file')
    else:
        nlu_file = os.path.join(env_config['interpreter_dir'],'training_data.json')

    app = create_app(port,env_config["server_ip"],env_config["server_port"],
                     env_config["conv_listen_address"],env_config["conv_api_port"],nlu_file,env_config["path_to_glove"])

    http_server = WSGIServer(('0.0.0.0', port), app)
      
    logging.basicConfig(level='DEBUG')
    logger.info(f"Server started at port {port}")
    http_server.serve_forever()
