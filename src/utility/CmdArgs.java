package utility;

import org.kohsuke.args4j.Option;

public class CmdArgs
{

	@Option(name = "-model", usage = "Specify model")
	public String model = "DMM";
	@Option(name = "-dataname", usage = "Data name")
	public String dataname = "SearchSnippets";

	@Option(name = "-corpus", usage = "Specify path to topic modeling corpus")
	public String corpus = "dataset/SearchSnippets_train.txt";

	@Option(name = "-generateCorpus", usage = "")
	public String generateCorpus = "";

	@Option(name = "-preTheta", usage = "")
	public String preTheta = "";

	@Option(name = "-preTopicAssignments", usage = "")
	public String preTopicAssignments = "";

	@Option(name = "-output", usage = "Specify path to save the results")
	public String output_dir = "results/";

	@Option(name = "-vectors", usage = "Specify path to the file containing word vectors")
	public String vectors = "";

	@Option(name = "-schema", usage = "Specify path to the file containing similarity")
	public String schema = "";

	@Option(name = "-MultiKEschema", usage = "Specify path to the file containing similarity")
	public String MultiKEschema = "";

	@Option(name = "-nlongdoc", usage = "Specify number of pseudo-long documents")
	public int nLongDoc = 1000;

	@Option(name = "-threshold", usage = "Specify threshold, unimportant correspondences between short texts and pseudo-documents are filtered")
	public double threshold = 0.001;

	@Option(name = "-localWC", usage = "A matrix of local word correlation in the original dataset in APU-DMM")
	public String localWC = "";

	@Option(name = "-GPUthreshold", usage = "Specify threshold in GPUDMM")
	public double GPUthreshold = 0.5;

	@Option(name = "-weight", usage = "Specify weight in GPUDMM")
	public double weight = 0.3;

	@Option(name = "-filterSize", usage = "Filter less semantic related word pairs in GPUDMM")
	public int filterSize = 20;

	@Option(name = "-ntopics", usage = "Specify number of topics")
	public int ntopics = 300;

	@Option(name = "-alpha", usage = "Specify alpha")
	public double alpha = 0.1;

	@Option(name = "-beta", usage = "Specify beta")
	public double beta = 0.1;

	@Option(name = "-gamma", usage = "Specify beta")
	public double gamma = 0.1;

	@Option(name = "-lambda", usage = "Specify mixture weight lambda")
	public double lambda = 0.6;

	@Option(name = "-niters", usage = "Specify number of iterations")
	public int niters = 1000;

	@Option(name = "-nBurnIn", usage = "Specify number of burnIn")
	public int nBurnIn = 500;

	@Option(name = "-twords", usage = "Specify number of top topical words")
	public int twords = 20;

	@Option(name = "-maxTd", usage = "Specify number of maximum topic in each document in GPU-PDMM (use 'tau' sympol in original paper)" )
	public int maxTd = 3;

	@Option(name = "-searchTopK", usage = " for Searching Space Pruning in GPU-PDMM.")
	public int searchTopK = 10;

	@Option(name = "-window", usage = "Specify the size of windon in WNTM method")
	public int window = 10;

	@Option(name = "-name", usage = "Specify a name to topic modeling experiment")
	public String expModelName = "MKGEGPU_thres0%d_%d";

	@Option(name = "-initFile")
	public String initTopicAssgns = "";

	@Option(name = "-sstep")
	public int savestep = 0;

	@Option(name = "-dir")
	public String dir = "";

	@Option(name = "-label")
	public String labelFile = "";

	@Option(name = "-prob")
	public String prob = "";

	@Option(name = "-topWords")
	public String topWords = "";

	@Option(name = "-paras", usage = "Specify path to hyper-parameter file")
	public String paras = "";

	@Option(name = "-DLDA_weights")
	public double DoubleLDA_weights = 1.;

	@Option(name = "-topWordsDir")
	public String topWordsPath = "";

	@Option(name = "-vocabFile")
	public String vocabFile = "";

	@Option(name = "-wikiDir")
	public String wikiDir = "";

	@Option(name = "-topicCoherEval")
	public String topicCoherEval = "npmi";

	@Option(name = "-topTC")
	public int topTC = -1;

}
