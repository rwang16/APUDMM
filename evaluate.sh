# Evaluate topic coherence and topic unique

project_path=$(pwd)
data_path="./datasets"
result_path="./results"

# Tweet, GoogleNews, SearchSnippets
dataset=SearchSnippets

topicCoherence_metric="pmi"
wiki_dir="${data_path}/${dataset}/word_wiki"
vocab_file="${data_path}/${dataset}/${dataset}_vocab.txt"

K=50
topWords_path="${result_path}/${dataset}/${dataset}_${K}/"
java -jar ${project_path}/out/artifacts/apudmm.jar\
           -model "TopicQualityEval"\
           -ntopics ${K}\
           -topWordsDir ${topWords_path}\
           -vocabFile ${vocab_file}\
           -wikiDir ${wiki_dir}\
           -topicCoherEval ${topicCoherence_metric}\
           -topTC 10

K=100
topWords_path="${result_path}/${dataset}/${dataset}_${K}/"
java -jar ${project_path}/out/artifacts/apudmm.jar\
           -model "TopicQualityEval"\
           -ntopics ${K}\
           -topWordsDir ${topWords_path}\
           -vocabFile ${vocab_file}\
           -wikiDir ${wiki_dir}\
           -topicCoherEval ${topicCoherence_metric}\
           -topTC 10