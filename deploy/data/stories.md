## happy path
* greet
  - utter_greet
* predict_delay
  - action_predict_delay
 
## New Story
* predict_delay_complete{"Route":"501","Direction":"e"}
  - slot{"Route":"501"}
  - slot{"Direction":"e"}
  - action_predict_delay_complete

## New Story
* predict_delay_complete{"Route":"506","hour":"16","day":"today"}
  - slot{"Route":"506"}
  - slot{"hour":"16"}
  - slot{"day":"today"}
  - action_predict_delay_complete


## New Story
* predict_delay_complete{"Route":"508","delta":"3","scale":"hours"}
  - slot{"Route":"508"}
  - slot{"delta":"3"}
  - slot{"scale":"hours"}
  - action_predict_delay_complete

## say goodbye
* goodbye
  - utter_goodbye
