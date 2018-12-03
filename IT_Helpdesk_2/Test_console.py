
from rasa_core.run import main


# In[1]:
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# In[3]:



# In[4]:


from rasa_core.domain import OntologyDomain
import os
path = './'
domain = OntologyDomain.load(os.path.join(path, "domain.json"),
                                     None)


from idare import KerasIDarePolicy
from rasa_core.featurizers import FloatSingleStateFeaturizer,MaxHistoryTrackerFeaturizer
feat=MaxHistoryTrackerFeaturizer(FloatSingleStateFeaturizer(),max_history=2)
policy = KerasIDarePolicy(feat)



from rasa_core.policies.ensemble import SimplePolicyEnsemble
ensemble = SimplePolicyEnsemble([policy])

# In[5]:


from rasa_core.agent import Agent
from suprath_nlu.wrapper import Interpreter
from rasa_core.tracker_store import InMemoryTrackerStore
_interpreter = Interpreter('./nlu_model/default/current','./nlu_model/default/current','C:/task/rasaextension/coreextension/examples/glove.6B/glove.6B.50d.txt')
#_tracker_store = InMemoryTrackerStore(domain)
_tracker_store = Agent.create_tracker_store(None,domain,None)

#print(_tracker_store)
#print(_tracker_store.red)

# In[7]:


agent = Agent(domain, ensemble, _interpreter, _tracker_store)

# Training
training_data = agent.load_data('./stories.md')

agent.train(training_data, epochs = 50)
# In[9]:


from rasa_core.run import create_input_channel
from rasa_core.channels.file import FileInputChannel
#usr_channel=FileInputChannel('./testio.txt')


# In[9]:


from rasa_core.channels.console import ConsoleInputChannel
import numpy as np
sender_id = str(123123213+np.random.randint(1000))
print("creating sender id"+sender_id)
usr_channel = ConsoleInputChannel(sender_id = sender_id)
ha_channel = None


# In[10]:


usr_channel


# In[11]:



logging.basicConfig(level=logging.DEBUG)
agent.handle_dual_channels([usr_channel],ha_channel)


# In[ ]:


interpreter.parse('My account is not active')

