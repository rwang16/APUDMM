# Run APUDMM

project_path=$(pwd)
data_path="./datasets"
result_path="./results"

algorithm=APUDMM
beta=0.01
# Tweet, GoogleNews : 0.8; SearchSnippets : 0.7
dataset=GoogleNews
threshold=0.8

K=50
alpha=1.0
for times in {1..5}; do
  java -jar ${project_path}/out/artifacts/apudmm.jar\
          -model ${algorithm}\
          -dataname ${dataset}\
          -alpha ${alpha}\
          -beta ${beta}\
          -ntopics ${K}\
          -schema ${data_path}/${dataset}/${dataset}_Word2VecSim.txt\
          -GPUthreshold ${threshold}\
          -corpus ${data_path}/${dataset}/${dataset}.txt\
          -output ${result_path}/${dataset}/${dataset}_${K}/\
          -name ${algorithm}_${alpha}_${beta}_${threshold}_${times}\
          -nBurnIn 500\
          -niters 1000
done

K=100
alpha=0.5
for times in {1..5}; do
  java -jar ${project_path}/out/artifacts/apudmm.jar\
          -model ${algorithm}\
          -dataname ${dataset}\
          -alpha ${alpha}\
          -beta ${beta}\
          -ntopics ${K}\
          -schema ${data_path}/${dataset}/${dataset}_Word2VecSim.txt\
          -GPUthreshold ${threshold}\
          -corpus ${data_path}/${dataset}/${dataset}.txt\
          -output ${result_path}/${dataset}/${dataset}_${K}/\
          -name ${algorithm}_${alpha}_${beta}_${threshold}_${times}\
          -nBurnIn 500\
          -niters 1000
done