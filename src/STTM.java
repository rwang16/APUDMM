
import eval.TopicQualityEval;
import models.*;


import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;

import utility.CmdArgs;
import eval.ClusteringEval;
import eval.ClassificationEval;

/**
 * STTM: A Java package for the short text topic models including DMM, BTM, WNTM, PTM, SATM, LDA, LFDMM, LFLDA, etc.
 *
 *
 * @author: Jipeng Qiang
 *
 * @version: 1.0
 *
 */
public class STTM
{
    public static void main(String[] args)
    {

        CmdArgs cmdArgs = new CmdArgs();
        CmdLineParser parser = new CmdLineParser(cmdArgs);
        try {

            parser.parseArgument(args);

            if (cmdArgs.model.equals("APUDMM")) {
                APUDMM apudmm = new APUDMM(cmdArgs.corpus, cmdArgs.output_dir,cmdArgs.vectors, cmdArgs.schema,
                        cmdArgs.GPUthreshold, cmdArgs.filterSize, cmdArgs.ntopics, cmdArgs.alpha, cmdArgs.beta,
                        cmdArgs.niters,cmdArgs.nBurnIn, cmdArgs.twords, cmdArgs.expModelName);
                apudmm.inference();
            }
            else if (cmdArgs.model.equals("ClusteringEval")) {
                ClusteringEval.evaluate(cmdArgs.labelFile, cmdArgs.dir,
                        cmdArgs.prob);
            }else if (cmdArgs.model.equals("ClassificationEval")) {
                ClassificationEval.evaluate(cmdArgs.labelFile, cmdArgs.dir,
                        cmdArgs.prob);
            }else if (cmdArgs.model.equals("TopicQualityEval")) {
                TopicQualityEval topicQualityEval = new TopicQualityEval(cmdArgs.ntopics, cmdArgs.topWordsPath,
                        cmdArgs.vocabFile, cmdArgs.wikiDir, cmdArgs.topTC, cmdArgs.topicCoherEval);
                topicQualityEval.computeTopicQualityEval();
            }
            else {
                System.out
                        .println("Error: Option \"-model\" must get \"LDA\" or \"DMM\" or \"BTM\" or \"WNTM\" or \"SATM\" or \"GPUDMM\" or \"GPU_PDMM\" or \"LDALDA\" or \"LFDMM\" or \"Eval\"");
                System.out
                        .println("\tLDA: Specify the Latent Dirichlet Allocation topic model");
                System.out
                        .println("\tDMM: Specify the one-topic-per-document Dirichlet Multinomial Mixture model");
                System.out
                        .println("\tBTM: Infer topics for Biterm");
                System.out
                        .println("\tWNTM: Infer topics for WNTM");
                System.out
                        .println("\tSATM: Infer topics using SATM");
                System.out
                        .println("\tPTM: Infer topics using PTM");
                System.out
                        .println("\tGPUDMM: Infer topics using GPUDMM");
                System.out
                        .println("\tGPU_PDMM: Infer topics using GPU_PDMM");
                System.out
                        .println("\tLFLDA: Infer topics using LFLDA");
                System.out
                        .println("\tLFDMM: Infer topics using LFDMM");
                System.out
                        .println("\tLDAinf: Infer topics for unseen corpus using a pre-trained LDA model");
                System.out
                        .println("\tDMMinf: Infer topics for unseen corpus using a pre-trained DMM model");
                System.out
                        .println("\tEval: Specify the document clustering evaluation");
                help(parser);
                return;
            }
        }
        catch (CmdLineException cle) {
            System.out.println("Error: " + cle.getMessage());
            help(parser);
            return;
        }
        catch (Exception e) {
            System.out.println("Error: " + e.getMessage());
            e.printStackTrace();
            return;
        }

        System.out.println("end!!!!!!!");
    }

    public static void help(CmdLineParser parser)
    {
        System.out
                .println("java -jar jSTTM.jar [options ...] [arguments...]");
        parser.printUsage(System.out);
    }
}
