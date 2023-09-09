package eval;

import java.io.*;
import java.util.*;

public class TopicQualityEval {
	public String topWordsPath;
    public int numTopic;
	public String corpusVocabFile;
	public String wikiDir;

	public int[] topTC;
    public int topTU;

	public String evaluation;
//	public static final int wiki_docs = 4776093;
	public final int numWikiDocs = 300000;

    private int[][] topicTopWords;


    public TopicQualityEval(int inNumTopic, String inTopWordsPath, String inCorpusVocabFile, String inWikiDir,
                            String inEvaluation) throws IOException {
        this(inNumTopic, inTopWordsPath, inCorpusVocabFile, inWikiDir,
                -1, 15, inEvaluation);
    }

    public TopicQualityEval(int inNumTopic, String inTopWordsPath, String inCorpusVocabFile, String inWikiDir,
                            int inTopTC, String inEvaluation) throws IOException {
        this(inNumTopic, inTopWordsPath, inCorpusVocabFile, inWikiDir,
                inTopTC, 15, inEvaluation);
    }

	public TopicQualityEval(int inNumTopic, String inTopWordsPath, String inCorpusVocabFile, String inWikiDir,
							  int inTopTC, int inTopTU, String inEvaluation) throws IOException{
        numTopic = inNumTopic;
        topWordsPath = inTopWordsPath;
		corpusVocabFile = inCorpusVocabFile;
		wikiDir = inWikiDir;
        if (inTopTC == -1){
            topTC = new int[]{5,10,15};
        } else {
            topTC = new int[]{inTopTC};
        }
        topTU = inTopTU;
		evaluation = inEvaluation;

	}

    /**
     * Compute topic coherence and topic unique
     * @throws IOException
     */
    public void computeTopicQualityEval() throws IOException {
        // Find topWords file
        List<String> topWordsFiles = findTopWordsFiles(topWordsPath);
        if (topWordsFiles.isEmpty()) {
            System.out.println("No 'topWords' files found in " + topWordsPath);
        } else {
            // if found, compute tc and tu
            int numExperience = topWordsFiles.size();
            double[] tc = new double[numExperience];
            double[] tu = new double[numExperience];
            double meanTC = 0.0;
            double meanTU = 0.0;
            for (int i = 0; i < numExperience; i++) {
                readTopWords(topWordsPath + "/" + topWordsFiles.get(i));
                tc[i] = Double.parseDouble(computeTopicCoherence());
                meanTC += tc[i];
                tu[i] = Double.parseDouble(computeTopicUnique());
                meanTU += tu[i];
            }

            // Print the information
            System.out.println("\nTopWords path : " + topWordsPath);
            System.out.println("The number of topic : " + numTopic);
            System.out.println("The number of experience: " + numExperience);
            System.out.print("Topic Coherence ("+evaluation+") : ");
            for (int i = 0; i < numExperience; i++) {
                System.out.print(tc[i] + " ");
            }
            System.out.print("\nTopic Unique : ");
            for (int i = 0; i < numExperience; i++) {
                System.out.print(tu[i] + " ");
            }
            System.out.println("\n\nThe mean of topic coherence : " + meanTC / numExperience);
            System.out.println("The mean of topic unique : " + meanTU / numExperience);
        }
    }

    /**
     * Read the vocabulary of the corpus
     * @param corpusPath
     * @throws IOException
     */
    private Map<String, Integer> readCorpusVocab(String corpusPath) throws IOException {
        Map<String, Integer> word2Id = new HashMap<>();
        BufferedReader reader = new BufferedReader(new FileReader(corpusPath));
        int wordIndex = 0;
        for (String word; (word = reader.readLine()) != null ;) {
            if (word.trim().length() == 0){
                continue;
            }
            word2Id.put(word, wordIndex);
            wordIndex += 1;
        }
        reader.close();
        return word2Id;
    }

    /**
     * Read the topWords in topWordsFile
     * @param topWordsFile
     * @throws IOException
     */
    private void readTopWords(String topWordsFile) throws IOException {
        Map<String, Integer> word2Id = readCorpusVocab(corpusVocabFile);
        topicTopWords = new int[numTopic][20];
        BufferedReader reader = new BufferedReader(new FileReader(topWordsFile));
        int lineIndex = 0;
        String line = null;
        int wordId = -1;
        while ((line = reader.readLine()) != null) {
            String[] items = line.trim().split(" ");
            // If there are English words in the topWords file, convert them to wordId using vocabulary
            for (int i = 0; i < items.length; i++) {
                if (isNumber(items[0])){
                    wordId = Integer.parseInt(items[i]);
                } else {
                    wordId = word2Id.get(items[i]);
                }
                topicTopWords[lineIndex][i] = wordId;
            }
            lineIndex += 1;
        }
        reader.close();
    }

    public boolean isNumber(String str){
        boolean ans = false;
        String pattern = "[+-]?((\\d+\\.?\\d*){1}|(\\d+\\.?){1}|(\\.{1}\\d+){1}){1}([eE]{1}[+-]?\\d+)?";
        if(str.matches(pattern)) {
            ans = true;
        }
        return ans;
    }

    /**
     * Read the information of Wiki
     * @param path
     * @return
     * @throws IOException
     */
    private Set<String> getWordWikis(String path) throws IOException{
        BufferedReader reader = new BufferedReader(new FileReader(path));
        Set<String> wordWikis = new HashSet<>();
        for (String line; (line = reader.readLine()) != null ;) {
            if (line.trim().length() == 0){
                continue;
            }
            wordWikis.add(line);
        }
        reader.close();
        return wordWikis;
    }

    /**
     * Compute the mean of topic coherence of all of topic
     * @return
     * @throws IOException
     */
    private String computeTopicCoherence() throws IOException {
        double totalTopicCoherence = 0.0;
        for (int i = 0; i < topTC.length; i++) {
            int top = topTC[i];
            for (int j = 0; j < topicTopWords.length; j++) {
                double topicQuality = topicCoherence(topicTopWords[j], top);
                totalTopicCoherence += topicQuality;
            }
        }
        double meanTC = totalTopicCoherence / (numTopic * topTC.length);
        return String.format("%.3f", meanTC);
    }

    /**
     * Compute topic coherence of a topic
     * @param topicK
     * @return
     * @throws IOException
     */
    private double topicCoherence(int[] topicK, int top) throws IOException {

        // Load the wiki information of the topWords
        Map<Integer, Set<String>> wikiInfo = new HashMap<>();
        // Compute the word term frequency
        double[] wordTF = new double[top];
        // Count the number of times two words co-occurrence
        double[][] wordCount = new double[top][top];
        for (int i = 0; i < top; i++) {
            int wordId = topicK[i];
            Set<String> wordInfo = getWordWikis(wikiDir + "/" + wordId);
            wordTF[i] = wordInfo.size();
            wikiInfo.put(wordId, wordInfo);
        }
        for (int i = 0; i < top; i++) {
            for (int j = i + 1; j < top; j++) {
                if (i != j){
                    Set<String> word1Info = wikiInfo.get(topicK[i]);
                    Set<String> word2Info = wikiInfo.get(topicK[j]);
                    for (String wiki : word1Info){
                        if (word2Info.contains(wiki)){
                            wordCount[i][j] += 1.0;
                        }
                    }
                }
            }
        }

        // Compute the topic coherence in topicK
        double tc = 0.0;
        for (int i = 0; i < top; i++) {
            for (int j = i + 1; j < top; j++) {
                // pi : the term frequency of word topicK[i]
                double pi = wordTF[i] / numWikiDocs;
                // pj : the term frequency of word topicK[j]
                double pj = wordTF[j] / numWikiDocs;
                // pij : the probability of word topicK[i] and topicK[j] co-occurrence
                double pij = wordCount[i][j] / numWikiDocs;
                if (pi==0 || pj==0 || pij == 0){
                    tc += 0.0;
                } else{
                    // Compute pmi or npmi as topic coherence
                    if ("pmi".equals(evaluation)){
                        tc += Math.log(pij / (pi * pj));
                    } else if ("npmi".equals(evaluation)){
                        tc += Math.log(pij / (pi * pj)) / (-Math.log(pij));
                    }
                }
            }
        }
        // The result is the mean of topic coherence of all the word-pairs
        tc = 2 * tc / (top * (top - 1));
        return tc;
    }

    /**
     * Compute topic unique
     * @return
     */
    private String computeTopicUnique() {
        // Count the num of word occurrence
        Map<Integer, Integer> cnt = new HashMap<>();
        for (int i = 0; i < topicTopWords.length; i++) {
            for (int j = 0; j < topTU; j++) {
                int wordId = topicTopWords[i][j];
                if (cnt.containsKey(wordId)){
                    cnt.put(wordId, cnt.get(wordId)+1);
                } else{
                    cnt.put(wordId, 1);
                }
            }
        }

        // Compute the mean of TU of all word
        double tu = 0.0;
        for (int i = 0; i < topicTopWords.length; i++) {
            for (int j = 0; j < topTU; j++) {
                tu += 1.0 / cnt.get(topicTopWords[i][j]);
            }
        }
        tu = tu / (numTopic * topTU);
        // Retain three significant digits
        return String.format("%.3f", tu);
    }

    /**
     * Find topWords file
     * @param path
     * @return
     */
    public static List<String> findTopWordsFiles(String path) {
        List<String> topWordsFiles = new ArrayList<>();
        File folder = new File(path);

        if (folder.exists() && folder.isDirectory()) {
            File[] files = folder.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isFile() && file.getName().endsWith(".topWords")) {
                        topWordsFiles.add(file.getName());
                    }
                }
            }
        }
        return topWordsFiles;
    }

    public static void main(String[] args) throws IOException{

        TopicQualityEval tqe = new TopicQualityEval(50,
                "D:\\TopicModel\\results\\SearchSnippets\\APUDMM\\APUDMM_50_1.0_0.01_0.7",
                "D:\\TopicModel\\dataset\\Short\\shortTextCorpus\\SearchSnippets\\SearchSnippets_vocab2.txt",
                "D:\\TopicModel\\dataset\\Short\\shortTextCorpus\\SearchSnippets\\word_wiki",10,"pmi");
        tqe.computeTopicQualityEval();
    }

}
