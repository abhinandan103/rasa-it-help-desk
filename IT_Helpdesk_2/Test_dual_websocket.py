
from rasa_core.run import main



# In[1]:
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# In[3]:



# In[4]:


from rasa_core.domain import OntologyDomain
from rasa_core.envconfig  import EnvConfig
import os
path = './'
domain = OntologyDomain.load(os.path.join(path, "domain.json"),
                                     None)


env_config = EnvConfig(os.path.join(path, "envconfig.json"))
from idare import KerasIDarePolicy
from rasa_core.featurizers import FloatSingleStateFeaturizer,MaxHistoryTrackerFeaturizer
feat=MaxHistoryTrackerFeaturizer(FloatSingleStateFeaturizer(),max_history=1)
policy = KerasIDarePolicy(feat)



from rasa_core.policies.ensemble import SimplePolicyEnsemble
ensemble = SimplePolicyEnsemble([policy])

# In[5]:


from rasa_core.agent import Agent
from rasa_core.interpreter import NaturalLanguageInterpreter
from rasa_core.tracker_store import InMemoryTrackerStore
_interpreter = NaturalLanguageInterpreter.create('./nlu_model/current')
_tracker_store = Agent.create_tracker_store(None,domain,env_config)



# In[7]:


agent = Agent(domain, ensemble, _interpreter, _tracker_store,env_config)


# In[9]:

from rasa_core.run import create_input_channel
from rasa_core.channels.file import FileInputChannel
#usr_channel=FileInputChannel('./testio.txt')


# In[9]:


from rasa_core.run import create_input_channel
from rasa_core.channels.custom_websocket import CustomInput
input_component1=CustomInput(None)
input_component2=CustomInput(None)
from rasa_core.channels.websocket import WebSocketInputChannel
usr_channel = WebSocketInputChannel(int(env_config.userchannelport),None,input_component1,http_ip='0.0.0.0')
ha_channel = WebSocketInputChannel(int(env_config.hachannelport),None,input_component2,http_ip='0.0.0.0')
usr_channel.output_channel = input_component1.output_channel
ha_channel.output_channel = input_component2.output_channel


# In[10]:


usr_channel


# In[11]:



logging.basicConfig()
agent.handle_dual_channels(usr_channel,ha_channel)


# In[ ]:


