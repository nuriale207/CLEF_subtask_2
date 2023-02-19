import argparse
import json
from pathlib import Path

import pandas as pd
from collections import Counter
from utils import transform_data, graphics
from utils.data_reader import json_reader, all_csv_reader, process_data, user_csv_reader
from utils.generate_topics import createPLDA, obtenerVectorTopics
from utils.preprocess_data import vocab_size, preprocesado
from utils.topics_metrics import sentiment_score


def create_arguments():
    parser = argparse.ArgumentParser(
        description='This runnable preproccesses the documents and generates a json file for each user with the '
                    'predicted topics in each post. It also generates a json file with the most important words in each '
                    'topic and their weight.')

    parser.add_argument('--csv', dest='csv_file', nargs='+',
                        help='Path to the csv file with the users posts information.')
    parser.add_argument('--labels', dest='csv_labels', nargs='+',
                        help='Path to the csv file with the users labels.')
    parser.add_argument('-k', dest='topics_per_label', nargs='+', default=1,
                        help='Indicates the amount of topics per label to generate')
    parser.add_argument('-n', dest='topic_words',  nargs='+', default=1,
                        help='Indicates the amount of words to include in the topics information file')
    parser.add_argument('--path',dest='path_dir', nargs='+',
                        help='Path to the directory to save the generated plda model and json files.')
    parser.add_argument('--generate', action="store_true",
                        help="Set this flag to generate topics for the documents")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = create_arguments()
    posts_dir = args.csv_file[0]
    labels_dir = args.csv_labels[0]
    num_topics=args.topics_per_label[0]
    print(num_topics)
    num_topics=int(num_topics)
    print("Num topics per label: "+str(num_topics))

    predict=args.generate
    dest_dir=args.path_dir[0]
    n=args.topic_words[0]
    print(n)
    posts_info=pd.read_csv(posts_dir)
    posts=list(posts_info["text"])

    labels_info=pd.read_csv(labels_dir)
    labels=list(posts_info["label"])
    labels[30]=1
    labels[200]=1
    labels[20]=1

    labels=[str(label) for label in labels]
    print(Counter(labels))

    preprocessed_posts=preprocesado(posts,True,True,True)




    vocab,vocab_n=vocab_size(preprocessed_posts)

    alpha=50/int(num_topics)*(len(set(labels)))
    eta=200/vocab_n


    plda_model=createPLDA(0,0,int(num_topics),alpha,eta,preprocessed_posts,labels)

    # all_users_topics=dict()
    # for user in users_json:
    #     df=user_csv_reader(user,users_json,time_dir)
    #     df,preprocessed_docs=process_data(df)
    #     lista_topics=[]
    #     for i in range(len(preprocessed_docs)):
    #         topics=obtenerVectorTopics(plda_model,preprocessed_docs[i])
    #         lista_topics.append(list(topics))
    #         # lista_topics.append({"post_id":df["postid"][i],"topics":list(topics)})
    #     df_topics=pd.DataFrame(data={"postid":df["postid"],"date":df["date"],"label":df["label"],"topics":lista_topics})
    #     filepath = Path(dest_dir+'/'+user+"_topics.tsv")
    #     filepath.parent.mkdir(parents=True, exist_ok=True)
    #     df_topics.to_csv(filepath,sep='\t')

    plda_model.save(dest_dir+'/plda_model.pkl')

    infoTopics = ""
    j = 0
    topic_label_dict=dict()
    topics_sentiment_score=[]
    for i in range(len(plda_model.topic_label_dict)):
        l = 0
        while (l < plda_model.topics_per_label and j < plda_model.k):
            infoTopics = infoTopics + plda_model.topic_label_dict[i] + ": "
            tuplas = plda_model.get_topic_words(j, int(n))
            topic_sentiment_score=(sentiment_score(tuplas))
            for k in range(len(tuplas)):
                tupla = tuplas[k]
                palabra = tupla[0]
                probabilidad = str(tupla[1])
                infoTopics = infoTopics + palabra + "," + probabilidad + "\t"
            infoTopics = infoTopics + "\n"
            topic_label_dict[j]={"label":plda_model.topic_label_dict[i],"sentiment_score":topic_sentiment_score}
            j += 1
            l += 1


    if predict:
        plda_model.train(500)
        lista_topics = []
        for post in preprocessed_posts:
            if len(post) == 0:
                post = ["blank"]
            topics = obtenerVectorTopics(plda_model, post)
            lista_topics.append(list(topics))

        df_topics = pd.DataFrame(lista_topics)
        df_topics_2 = pd.DataFrame(data={"id": posts_info["id"]})
        df_topics = pd.concat([df_topics_2, df_topics], axis=1)

        filepath = Path(dest_dir + '/' + posts_dir + "_topics.tsv")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        df_topics.to_csv(filepath, sep='\t')

        plda_model.save(dest_dir + '/plda_model.pkl')



    with open(dest_dir + '/topic_words.txt', 'w') as outfile:
        outfile.write(infoTopics)

    topics = infoTopics.split("\n")

    json_topic_label_dict=json.dumps(topic_label_dict)
    with open(dest_dir + '/topic_label_dict.json', 'w') as outfile:
        outfile.write(json_topic_label_dict)
    i = 0
    for topic in topics:
        dic = {}

        if topic == "":
            break

        if topic.__contains__(':'):
            topic = topic.split(':')[1]

        dic = transform_data.lineToDict(topic)
        graphics.crearNubesPalabras(dic, True, dest_dir + "/" + str(i))
        i = i + 1