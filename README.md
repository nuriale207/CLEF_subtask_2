# CLEF_subtask_2_vectors

## In this workspace are the runnable and files needed to generate the topics and topics distributions in post level

### SCRIPT: PLDA model inference

In order to generate the PLDA model run generate_plda_model.py:

	usage: generate_plda_model.py [-h] [--csv CSV_FILE [CSV_FILE ...]] [--labels CSV_LABELS [CSV_LABELS ...]] [-k TOPICS_PER_LABEL [TOPICS_PER_LABEL ...]] [-n TOPIC_WORDS [TOPIC_WORDS ...]] [--path PATH_DIR [PATH_DIR ...]]
                              [--generate]

    This runnable preproccesses the documents and generates a json file for each user with the predicted topics in each post. It also generates a json file with the most important words in each topic and their weight.

    options:
    -h, --help            show this help message and exit
    --csv CSV_FILE [CSV_FILE ...]
                        Path to the csv file with the users posts information.
    --labels CSV_LABELS [CSV_LABELS ...]
                       Path to the csv file with the users labels.
    -k TOPICS_PER_LABEL [TOPICS_PER_LABEL ...]
                    Indicates the amount of topics per label to generate
    -n TOPIC_WORDS [TOPIC_WORDS ...]
                    Indicates the amount of words to include in the topics information file
    --path PATH_DIR [PATH_DIR ...]
                    Path to the directory to save the generated plda model and json files.
    --generate            Set this flag to generate topics distribution for the documents

### SCRIPT: Topic distribution prediction for test posts

The runnable gets a one column CSV file with just the posts in that column and returns a CSV file with n + 1 topics, being n the number of topics

    usage: predict_topics.py [-h] [--csv CSV_FILE [CSV_FILE ...]] [--model MODEL [MODEL ...]] [--path PATH_DIR [PATH_DIR ...]]

    This runnable preproccesses the documents and generates a json file for each user with the predicted topics in each post. It also generates a json file with the most important words in each topic and their weight.
    
    options:
      -h, --help            show this help message and exit
      --csv CSV_FILE [CSV_FILE ...]
                            Path to the csv file with the users posts information.
      --model MODEL [MODEL ...]
                            Path to the PLDA model.
      --path PATH_DIR [PATH_DIR ...]
                            Path to the directory to save the generated files.


### USEFUL CODE SNIPPETS

Code snipet to preprocess one post and infer it's topic distribution given a PLDA model path

    from utils.generate_topics import obtenerVectorTopics, load_PLDA_model
    from utils.preprocess_data import preprocesado

    plda_model=load_PLDA_model(model_path)
    preprocessed_text=preprocesado_post(post,True,True,True)
    topics=obtenerVectorTopics(plda_model,post)


