import argparse
from pathlib import Path
import time
import pandas as pd

from utils.generate_topics import obtenerVectorTopics, load_PLDA_model
from utils.preprocess_data import preprocesado


def create_arguments():
    parser = argparse.ArgumentParser(
        description='This runnable preproccesses the documents and generates a json file for each user with the '
                    'predicted topics in each post. It also generates a json file with the most important words in each '
                    'topic and their weight.')

    parser.add_argument('--csv', dest='csv_file', nargs='+',
                        help='Path to the csv file with the users posts information.')
    parser.add_argument('--model', dest='model', nargs='+',
                        help='Path to the model.')


    parser.add_argument('--path',dest='path_dir', nargs='+',
                        help='Path to the directory to save the generated files.')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = create_arguments()
    posts_dir = args.csv_file[0]
    model_path=args.model[0]

    dest_dir=args.path_dir[0]
    posts_info = pd.read_csv(posts_dir, sep="\t", index_col=False,header=None)
    # posts=list(posts_info["text"])
    posts = posts_info.values
    posts=[item for sublist in posts for item in sublist]
    time_0 = time.time()

    preprocessed_posts=preprocesado(posts,True,True,True)
    print("Time to preprocess 2000 documents: " + str(time.time() - time_0))

    time_0=time.time()
    plda_model=load_PLDA_model(model_path)

    print("Time to load the model: "+str(time.time()-time_0))
    #plda_model.train(500)

    time_0 = time.time()

    lista_topics=[]
    for post in preprocessed_posts:
        if len(post)==0:
            post=["blank"]
        topics=obtenerVectorTopics(plda_model,post)

        lista_topics.append(list(topics))
    print("Time to process 2000 documents: " + str(time.time() - time_0))

    df_topics=pd.DataFrame(lista_topics)
    # df_topics_2=pd.DataFrame(data={"id":posts_info["id"]})
    df_topics=pd.concat([posts_info, df_topics], axis=1)


    filepath = Path(dest_dir+'/'+posts_dir+"_topics.tsv")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df_topics.to_csv(filepath,sep='\t',header=False)





