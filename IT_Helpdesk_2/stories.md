## Password Reset 1
* greet             % {"methods":"none"}

## Password Reset 1
* inform{"password_reset": null}     % {"processes":"password_reset","methods":"byname"}

## Password Reset Two
* inform{"password_reset": null,"application_skype":null}     % {"usr_slots":{"application_skype":{"confidence":1.0}},"processes":"password_reset","methods":"byname"}

## Password Reset Three
* inform{"password_reset": null,"application_salesforce":null}     % {"usr_slots":{"application_salesforce":{"confidence":1.0}},"processes":"password_reset","methods":"byname"}

## Password Reset Four
* inform{"password_reset":null, "application_serviceacc":null}     % {"usr_slots":{"application_serviceacc":{"confidence":1.0}},"processes":"password_reset","methods":"byname"}

## Password Reset Five
* inform{"ID": "1234"}     % {"usr_slots":{"ID":{"confidence":1.0}}}

## Password Reset Six
- request{"ID":null} % {"sys_goals":{"requested_slot":{"value":"ID","confidence":1.0}}}


## Password Reset Six_1
- request{"ID":null} % {"sys_goals":{"requested_slot":{"value":"ID","confidence":1.0}}}
* inform{"ID": "1234"}     % {"usr_slots":{"ID":{"confidence":1.0}},"sys_goals":{"requested_slot":{"value":"ID","confidence":0.0}}}

## Password Reset Six_2
- request{"ID":null} % {"sys_goals":{"requested_slot":{"value":"ID","confidence":1.0}}}
* deny     % {"usr_slots":{"unavailable_slots":{"value":"ID","confidence":1.0}}}

## Password Reset Six_3
- confirm{"ID":null} 
* deny     % {"usr_slots":{"ID":{"confidence":0.0}}}

## Password Reset Seven
- request{"send_otp":null} % {"sys_goals":{"requested_slot":{"value":"send_otp","confidence":1.0}}}

## Password Reset Eight
- request{"send_otp":null} % {"sys_goals":{"requested_slot":{"value":"send_otp","confidence":1.0}}}
* affirm % {"usr_slots":{"send_otp":{"confidence":1.0}}}

## Password Reset Eight_den
- request{"send_otp":null} % {"sys_goals":{"requested_slot":{"value":"send_otp","confidence":1.0}}}
* deny % {"usr_slots":{"unavailable_slots":{"value":"send_otp","confidence":1.0}}}

## Password Reset 91
- request{"security_code":null} % {"sys_goals":{"requested_slot":{"value":"security_code","confidence":1.0}}}

## Password Reset 90
- request{"security_code":null} % {"sys_goals":{"requested_slot":{"value":"security_code","confidence":1.0}}}
* deny % {"usr_slots":{"unavailable_slots":{"value":"security_code","confidence":1.0}}}

## Password Reset 9
- request{"send_otp":null} % {"sys_goals":{"requested_slot":{"value":"send_otp","confidence":1.0}}}
* inform{"number":"1234"} % {"usr_slots":{"security_code":{"confidence":1.0}}}


## General 1 
- inform{"password_reset":null,"finished":null} % {"usr_slots":{"resolved":{"confidence":0.0}},"methods":"finished"}
* gratitude 

## General 2
- anythingelse 
* deny % {"methods":"bye"}
- bye % {"methods":"reset"}

