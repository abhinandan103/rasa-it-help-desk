from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import logging
import os
import json
import tempfile
import zipfile
from functools import wraps,partial

import pymysql
from builtins import str
from flask import Flask, request, abort, jsonify
from werkzeug.wrappers import Response
from flask_cors import CORS, cross_origin
from gevent.pywsgi import WSGIServer
from typing import Union, Text, Optional

from rasa_core import utils, events
from rasa_core.agent import Agent
from rasa_core.channels.direct import CollectingOutputChannel
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.tracker_store import TrackerStore
from rasa_core.trackers import DialogueStateTracker
from rasa_core.version import __version__
from rasa_core.domain import OntologyDomain
from rasa_core.channels.botframework import BotFramework,BotFrameworkInput
from rasa_core.channels.rest import HttpInputChannel
from rasa_core.utils import  read_json_file

import idare
from suprath_nlu.wrapper import Interpreter

from typing import Union
import typing




if typing.TYPE_CHECKING:
    from rasa_core.interpreter import NaturalLanguageInterpreter as NLI

logger = logging.getLogger(__name__)


def sql_feedback_logger(message):
    data = message.split(' ')
    host_ip = env_json['db_host']
    db_port = env_json['db_port']
    db_username = env_json['db_username']
    db_password = env_json['db_password']
    connection = pymysql.connect(host=host_ip,
                         user=db_username,
                         password=db_password,
                         db='agentdb',
                         charset='utf8mb4',
                         cursorclass=pymysql.cursors.DictCursor)
    try:
        with connection.cursor() as cursor:
            sql=("insert into feedback.feedback (conv_id,event_id,feedback_type,reward,trained_status)"
                f"values ('{data[0]}','{data[1]}','{data[2]}',{int(data[3])},{0})")
            cursor.execute(sql)
        connection.commit()
        logger.debug("Inserted feedback successfully")
    except pymysql.IntegrityError:
        logger.warn("failed to insert values in pymysql: {}".format(data))
    finally:
        connection.close()


def create_argument_parser():
    """Parse all the command line arguments for the server script."""

    parser = argparse.ArgumentParser(
            description='starts server to serve an agent')
    parser.add_argument(
            '-e', '--environment',
            type=str,
            help="environment config file to run with the server")
    parser.add_argument(
            '--cors',
            nargs='*',
            type=str,
            help="enable CORS for the passed origin. "
                 "Use * to whitelist all origins")
    parser.add_argument(
            '--auth_token',
            type=str,
            help="Enable token based authentication. Requests need to provide "
                 "the token to be accepted.")
    parser.add_argument(
            '-o', '--log_file',
            type=str,
            default="rasa_core.log",
            help="store log file in specified file")

    utils.add_logging_option_arguments(parser)
    return parser


def ensure_loaded_agent(agent):
    """Wraps a request handler ensuring there is a loaded and usable model."""

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            __agent = agent()
            if not __agent:
                return Response(
                        "No agent loaded. To continue processing, a model "
                        "of a trained agent needs to be loaded.",
                        status=503)

            return f(*args, **kwargs)

        return decorated

    return decorator


def bool_arg(name, default=True):
    # type: ( Text, bool) -> bool
    """Return a passed boolean argument of the request or a default.

    Checks the `name` parameter of the request if it contains a valid
    boolean value. If not, `default` is returned."""

    return request.args.get(name, str(default)).lower() == 'true'


def request_parameters():
    if request.method == 'GET':
        return request.args
    else:

        try:
            return request.get_json(force=True)
        except ValueError as e:
            logger.error("Failed to decode json during respond request. "
                         "Error: {}.".format(e))
            raise


def requires_auth(token=None):
    # type: (Optional[Text]) -> function
    """Wraps a request handler with token authentication."""

    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            provided = request.args.get('token')
            if token is None or provided == token:
                return f(*args, **kwargs)
            abort(401)

        return decorated

    return decorator


def _create_agent(
        model_directory,  # type: Text
        interpreter,  # type: Union[Text,NLI,None]
        action_factory=None,  # type: Optional[Text]
        tracker_store=None  # type: Optional[TrackerStore]
):
    # type: (...) -> Optional[Agent]
    try:

        return Agent.load(model_directory, interpreter,
                          tracker_store=tracker_store,
                          action_factory=action_factory)
    except Exception as e:
        logger.warn("Failed to load any agent model. Running "
                    "Rasa Core server with out loaded model now. {}"
                    "".format(e))
        return None


def create_app(env_json_file,
               loglevel="INFO",  # type: Optional[Text]
               logfile="rasa_core.log",  # type: Optional[Text]
               cors_origins=None,  # type: Optional[List[Text]]
               action_factory=None,  # type: Optional[Text]
               auth_token=None,  # type: Optional[Text]
               tracker_store=None  # type: Optional[TrackerStore]
               ):
    """Class representing a Rasa Core HTTP server."""

    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    # Setting up logfile
    utils.configure_file_logging(loglevel, logfile)

    if not cors_origins:
        cors_origins = []

    with open(env_json_file) as f:
        env_json = json.load(f)

    domain_file = env_json['domain']

    interpreter_file = env_json['interpreter_dir']
    path_to_glove = env_json['path_to_glove']


    tracker_store = tracker_store

    action_factory = action_factory

    

    # this needs to be an array, so we can modify it in the nested functions...
    domain = OntologyDomain.load(os.path.join(path, domain_file),
                                     None)


    
    from rasa_core.featurizers import FloatSingleStateFeaturizer,MaxHistoryTrackerFeaturizer
    feat=MaxHistoryTrackerFeaturizer(FloatSingleStateFeaturizer(),max_history=1)
    policy = idare.KerasIDarePolicy(feat)



    from rasa_core.policies.ensemble import SimplePolicyEnsemble
    ensemble = SimplePolicyEnsemble([policy])
    ensemble = ensemble.load(env_json['dst_model_dir'])

    # In[5]:


    from rasa_core.agent import Agent
    from rasa_core.interpreter import NaturalLanguageInterpreter
    from rasa_core.tracker_store import InMemoryTrackerStore
    _interpreter = Interpreter(suprath_dir = interpreter_file, rasa_dir = interpreter_file, 
                                path_to_glove = path_to_glove)
    logger.info("NLU interpreter loaded successfully")
    _tracker_store = Agent.create_tracker_store(None,domain,env_json)
    _agent = [Agent(domain, ensemble, _interpreter, _tracker_store,env_json)]
    global processor
    feedback_logger = sql_feedback_logger
    processor = _agent[0]._create_processor(feedback_logger=feedback_logger)
    
    usr_channel = None
    ha_channel = None
    teams_channel = None

    def agent():
        if _agent and _agent[0]:
            return _agent[0]
        else:
            return None

    @app.route("/",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    def hello():
        """Check if the server is running and responds with the version."""
        return "hello from Rasa Core: " + __version__

    @app.route("/version",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    def version():
        """respond with the version number of the installed rasa core."""

        return jsonify({'version': __version__})

    # <sender_id> can be be 'default' if there's only 1 client
    @app.route("/run",
               methods=['POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def continue_predicting():
        """Continue a prediction started with parse.

        Caller should have executed the action returned from the parse
        endpoint. The events returned from that executed action are
        passed to continue which will trigger the next action prediction.

        If continue predicts action listen, the caller should wait for the
        next user message."""
        from rasa_core.run import create_input_channel
        from rasa_core.channels.custom_websocket import CustomInput
        input_component1=CustomInput(None)
        input_component2=CustomInput(None)
        from rasa_core.channels.websocket import WebSocketInputChannel
        global usr_channel
        global ha_channel
        try:
            usr_channel = WebSocketInputChannel(int(env_json["userchannelport"]),None,input_component1,http_ip='0.0.0.0')

            botf_input_channel = BotFrameworkInput(
                  app_id=env_json['teams_app_id'],
                    app_password=env_json['teams_app_password']
                  )
            teams_channel = HttpInputChannel(int(env_json['userchannelport2']),'/webhooks/botframework',botf_input_channel)

        except OSError as e:
            logger.error(str(e))
            return str(e)


        usr_channel.output_channel = input_component1.output_channel
        teams_channel.output_channel = botf_input_channel.output_channel

        op_agent = agent()
        op_agent.handle_custom_processor([usr_channel,teams_channel],usr_channel,processor)

        return "ok"


    @app.route("/conversations/<sender_id>/tracker/events",
               methods=['POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def append_events(sender_id):
        """Append a list of events to the state of a conversation"""

        request_params = request.get_json(force=True)
        evts = events.deserialise_events(request_params)
        tracker = agent().tracker_store.get_or_create_tracker(sender_id)
        for e in evts:
            tracker.update(e)
        agent().tracker_store.save(tracker)
        return jsonify(tracker.current_state())

    @app.route("/conversations",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def list_trackers():
        return jsonify(list(agent().tracker_store.keys()))

    @app.route("/conversations/<sender_id>/tracker",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def retrieve_tracker(sender_id):
        """Get a dump of a conversations tracker including its events."""

        # parameters
        use_history = bool_arg('ignore_restarts', default=False)
        should_include_events = bool_arg('events', default=True)
        until_time = request.args.get('until', None)

        # retrieve tracker and set to requested state
        tracker = agent().tracker_store.get_or_create_tracker(sender_id)
        if until_time is not None:
            tracker = tracker.travel_back_in_time(float(until_time))

        # dump and return tracker
        state = tracker.current_state(
                should_include_events=should_include_events,
                only_events_after_latest_restart=use_history)
        return jsonify(state)

    @app.route("/conversations/<sender_id>/tracker",
               methods=['PUT', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def update_tracker(sender_id):
        """Use a list of events to set a conversations tracker to a state."""

        request_params = request.get_json(force=True)
        tracker = DialogueStateTracker.from_dict(sender_id,
                                                 request_params,
                                                 agent().domain)
        agent().tracker_store.save(tracker)

        # will override an existing tracker with the same id!
        agent().tracker_store.save(tracker)
        return jsonify(tracker.current_state(should_include_events=True))

    @app.route("/conversations/<sender_id>/parse",
               methods=['GET', 'POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def parse(sender_id):
        request_params = request_parameters()

        if 'query' in request_params:
            message = request_params.pop('query')
        elif 'q' in request_params:
            message = request_params.pop('q')
        else:
            return Response(
                    jsonify(error="Invalid parse parameter specified."),
                    status=400,
                    mimetype="application/json")

        try:
            # Fetches the predicted action in a json format
            response = agent().start_message_handling(message, sender_id)
            return jsonify(response)

        except Exception as e:
            logger.exception("Caught an exception during parse.")
            return Response(jsonify(error="Server failure. Error: {}"
                                          "".format(e)),
                            status=500,
                            content_type="application/json")

    @app.route("/conversations/<sender_id>/respond",
               methods=['GET', 'POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def respond(sender_id):
        request_params = request_parameters()

        if 'query' in request_params:
            message = request_params.pop('query')
        elif 'q' in request_params:
            message = request_params.pop('q')
        else:
            return Response(jsonify(error="Invalid respond parameter "
                                          "specified."),
                            status=400,
                            mimetype="application/json")

        try:
            # Set the output channel
            out = CollectingOutputChannel()
            # Fetches the appropriate bot response in a json format
            responses = agent().handle_message(message,
                                               output_channel=out,
                                               sender_id=sender_id)
            return jsonify(responses)

        except Exception as e:
            logger.exception("Caught an exception during respond.")
            return Response(jsonify(error="Server failure. Error: {}"
                                          "".format(e)),
                            status=500,
                            content_type="application/json")

    @app.route("/nlu_parse",
               methods=['GET', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def nlu_parse():

        request_params = request_parameters()

        if 'query' in request_params:
            message = request_params.get('query')
        elif 'q' in request_params:
            message = request_params.get('q')
        else:
            return Response(jsonify(error="Invalid respond parameter "
                                          "specified."),
                            status=400,
                            mimetype="application/json")
        return jsonify(_interpreter.parse(message))

    @app.route("/email/<sender_id>/respond",
               methods=['GET', 'POST', 'OPTIONS'])
    @cross_origin(origins=cors_origins)
    @requires_auth(auth_token)
    @ensure_loaded_agent(agent)
    def email(sender_id):
        request_params = request_parameters()

        if 'query' in request_params:
            message = request_params.get('query')
        elif 'q' in request_params:
            message = request_params.get('q')
        else:
            return Response(jsonify(error="Invalid respond parameter "
                                          "specified."),
                            status=400,
                            mimetype="application/json")

        global teams_channel
        inter_channel_mapper = read_json_file(env_json.get('inter_channel_mapper'))
        teams_channel.output_channel.id_map.update(inter_channel_mapper)
        #temporary code follows
        teams_channel.output_channel.id_map.update({sender_id:
                            inter_channel_mapper[list(inter_channel_mapper.keys())[0]]})
        teams_channel.output_channel.reverse_id_map.update({list(inter_channel_mapper.keys())[0]:sender_id})
        #temporary code ends

        email_id = request_params.get('email_id')
        preprocessor = partial(idare.email_preprocessor,email_id=email_id)
        try:
            # Set the output channel
            out = CollectingOutputChannel()
            # Fetches the appropriate bot response in a json format
            agent().handle_email(message, email_preprocessor = preprocessor,
                                                output_channel=out,
                                                alternate_channel = teams_channel,
                                               sender_id=sender_id)
            response = out.latest_output()

            return jsonify(response)

        except Exception as e:
            logger.exception("Caught an exception during respond.")
            return Response(jsonify(error="Server failure. Error: {}"
                                          "".format(e)),
                            status=500,
                            content_type="application/json")


    @app.route("/load_nlu", methods=['POST', 'OPTIONS'])
    @requires_auth(auth_token)
    @cross_origin(origins=cors_origins)
    def load_nlu_model():
        """Loads a zipped model, replacing the existing one."""

        if 'nlu_model' not in request.files:
            # model file is missing
            abort(400)

        model_file = request.files['nlu_model']

        logger.info("Received new nlu_model through REST interface.")
        zipped_path = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        zipped_path.close()

        model_file.save(zipped_path.name)

        logger.debug("Downloaded model to {}".format(zipped_path.name))

        zip_ref = zipfile.ZipFile(zipped_path.name, 'r')
        zip_ref.extractall(interpreter_file)
        zip_ref.close()
        logger.debug("Unzipped model to {}".format(
                os.path.abspath(interpreter_file)))
        
        global processor
        del processor.interpreter
        
        _interpreter.reload()

        processor.interpreter = _interpreter
        agent().interpreter = _interpreter
        logger.debug("Finished loading new interpreter.")
        return jsonify({'success': 1})

    @app.route("/load_idare", methods=['GET', 'OPTIONS'])
    @requires_auth(auth_token)
    @cross_origin(origins=cors_origins)
    def load_idare():
        """Reload idare."""
        from imp import reload
        global idare
        try:
            idare = reload(idare)
        except Exception as e:
            return str(e)
        return jsonify({'success': 1})
    return app


if __name__ == '__main__':
    # Running as standalone python application
    arg_parser = create_argument_parser()
    cmdline_args = arg_parser.parse_args()

    # Setting up the color scheme of logger
    utils.configure_colored_logging(cmdline_args.loglevel)

    # Setting up the rasa_core application framework

    path = './'
    env_json_file = os.path.join(path, cmdline_args.environment)
    
    with open(env_json_file) as f:
        env_json = json.load(f) 

    app = create_app(env_json_file,
                     cmdline_args.loglevel,
                     cmdline_args.log_file,
                     cmdline_args.cors,
                     auth_token=cmdline_args.auth_token)

    port = int(env_json['server_port'])
    


    # Running the server at 'this' address with the
    # rasa_core application framework
    http_server = WSGIServer(('0.0.0.0',port), app)
    logger.info("Started http server on port %s" % port)
    logger.info("Up and running")
    try:
        http_server.serve_forever()
    except Exception as exc:
        logger.exception(exc)

