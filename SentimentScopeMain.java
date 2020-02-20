package org.statnlp.example;

import gnu.trove.map.hash.TIntIntHashMap;
import gnu.trove.map.hash.TIntObjectHashMap;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.statnlp.commons.ml.opt.OptimizerFactory;
import org.statnlp.commons.types.Instance;
import org.statnlp.commons.types.Label;
import org.statnlp.hypergraph.DiscriminativeNetworkModel;
import org.statnlp.hypergraph.FeatureManager;
import org.statnlp.hypergraph.GlobalNetworkParam;
import org.statnlp.hypergraph.NetworkCompiler;
import org.statnlp.hypergraph.NetworkConfig;
import org.statnlp.hypergraph.NetworkModel;
import org.statnlp.hypergraph.StringIndex;
import org.statnlp.hypergraph.neural.BidirectionalLSTM;
import org.statnlp.hypergraph.neural.GlobalNeuralNetworkParam;
import org.statnlp.hypergraph.neural.NeuralNetworkCore;
import org.statnlp.sentiment.spanmodel.common.SpanModelGlobal;
import org.statnlp.targetedsentiment.common.TSFeatureValueProvider;
import org.statnlp.targetedsentiment.common.TSInstance;
import org.statnlp.targetedsentiment.common.TargetSentimentGlobal;
import org.statnlp.ui.visualize.type.VisualizationViewerEngine;
import org.statnlp.ui.visualize.type.VisualizeGraph;


public class SentimentScopeMain {
	
	public static String[] LABELS = new String[] {"Entity-positive", "Entity-neutral", "Entity-negative", "O"};
	
	public static int num_iter = 3000;
	public static int begin_index = 0;
	public static int end_index = 0;
	public static boolean mentionpenalty = false;
	public static int NEMaxLength = 7;
	public static int SpanMaxLength = 8;
	public static int numThreads = 20;
	public static double l2 = 0.0005;
	public static String embedding = "";
	public static int gpuId = -1;
	public static String neuralType =  "continuous";
	public static String nnOptimizer = "lbfgs";
	public static String nerOut = "nn-crf-interface/nlp-from-scratch/me/output/ner_out.txt";
	public static String neural_config = "nn-crf-interface/neural_server/neural.debug.config";
	public static boolean iobes = true;
	public static OptimizerFactory optimizer = OptimizerFactory.getLBFGSFactory();
	public static boolean DEBUG = false;
	public static String SEPERATOR = "\t";
	public static int evaluateEvery = 0;
	public static String additionalDataset = "none";
	public static String UNK = "<UNK>";
	public static int hiddenSize = 100;


	
	public static boolean SKIP_TRAIN = false;
	public static boolean SKIP_TEST = false;
	public static String in_path = "data//Twitter_";
	public static String out_path = "experiments//sentiment//models//<modelname>//Twitter_";
	public static String feature_file_path = in_path + "//feature_files//";
	public static boolean visual = false;
	public static String lang = "en";
	public static String embedding_suffix = ""; 
	public static boolean word_feature_on = true;
	public static String subpath = "default";
	public static String modelname = "sentimentspan_latent";
	public static NetworkModel model = null;
	
	public static void main(String args[]) throws IOException, InterruptedException{
		
		
		processArgs(args);
		
		if (additionalDataset.equals("wnut"))
		{
			TargetSentimentGlobal.LABELS = new String[] {"O", "B-positive", "I-positive", "B-neutral", "I-neutral", "B-negative", "I-negative", "B-unknown","I-unknown"};
		}
		
		TargetSentimentGlobal.init();
		
		NetworkConfig.L2_REGULARIZATION_CONSTANT = l2;
		NetworkConfig.NUM_THREADS = numThreads;
		
		System.out.println("#iter=" + num_iter + " L2=" + NetworkConfig.L2_REGULARIZATION_CONSTANT + " lang=" + lang + " modelname="+modelname );
		
		if (!embedding.equals(""))
		{
			TargetSentimentGlobal.ENABLE_WORD_EMBEDDING = true;
			
			if (embedding_suffix.equals("fasttext"))
			{
				TargetSentimentGlobal.EMBEDDING_WORD_LOWERCASE = true;
				UNK = "</s>";
			}
		}
		
		
		
		System.out.println("ENABLE_WORD_EMBEDDING=" + TargetSentimentGlobal.ENABLE_WORD_EMBEDDING);
		if (TargetSentimentGlobal.ENABLE_WORD_EMBEDDING)
		{
			TargetSentimentGlobal.initWordEmbedding(lang + "_" +embedding, hiddenSize);
			NetworkConfig.USE_NEURAL_FEATURES = true;
		}
		
		in_path = TargetSentimentGlobal.getInPath(modelname) + lang + "//";
		
		if (modelname.startsWith("baseline_pipeline"))
		{
			out_path = out_path.replace("<modelname>", "baseline_pipeline");
		}
		else
		{
			out_path = out_path.replace("<modelname>", modelname);
			
		}
		out_path = out_path + lang + "//" + subpath + "//";
		feature_file_path = in_path + "//feature_files//";
		TargetSentimentGlobal.feature_file_path = feature_file_path;
		
		File directory = new File(out_path);
		if (!directory.exists())
        {
            directory.mkdirs();
        }
		
		if (modelname.equals("baseline_pipelineSent"))
		{
			in_path = "experiments//sentiment//model//baseline_pipeline//Twitter_" + lang + "//temp//";
		}
		
		TargetSentimentGlobal.setLang(lang);


		for(int i = begin_index; i <= end_index; i++)
		{
			
			
			
			
			String train_file;
			String test_file;
			String dev_file;
			String model_file;
			String result_file;
			String iter = num_iter + "";
			String weight_push;
			String train_file_additional;
			
			
			System.out.println("Executing Data " + i);
			train_file = in_path + "train." + i +".coll";
			test_file = in_path + "test." + i + ".coll";
			dev_file = in_path + "dev." + i + ".coll";
			model_file = out_path + modelname + "." + i + ".model";
			result_file = out_path + "result." + i + ".out";
			weight_push = in_path + "weight0.data";
			train_file_additional = "data//" + additionalDataset +"//train";
			
			if (additionalDataset.startsWith("sst"))
			{
				train_file_additional = "data//sentiment//" + additionalDataset + "//train.ptb.txt";
			}
			
			System.out.println("Execute data " + i);
			
			if (modelname.equals("baseline_pipelineNE"))
			{
				train_file = in_path + "train." + i +".coll";
				test_file = in_path + "test." + i + ".coll";
				model_file = out_path + "3node." + i + ".model";
				result_file = out_path + "result." + i + ".NE.out";
			}
			else if (modelname.equals("baseline_pipelineSent"))
			{
				train_file = in_path + "train." + i +".pipeline.sent";
				test_file = in_path + "test." + i + ".pipeline.sent";
				model_file = out_path + "pipelineSent." + i + ".model";
				result_file = out_path + "result." + i + ".out";
			}
			
			TSInstance<Label>[] trainInstances = readCoNLLData(train_file, true, true);
			
			if (TargetSentimentGlobal.ENABLE_ADDITIONAL_DATA == true)
			{
				if (additionalDataset.startsWith("sst"))
				{
					trainInstances = readAdditionalSentimentData(trainInstances, train_file_additional, true, true);
				}
				else if (additionalDataset.equalsIgnoreCase("wnut"))
				{
					trainInstances = readAdditionalCoNLLData(trainInstances, train_file_additional, true, true);
				}
			}
			
			TSInstance<Label>[] testInstances = readCoNLLData(test_file, true, false);
			
			
			TSInstance<Label>[] devInstances = null;
			
			if (begin_index == 11) 
			{
				devInstances = readCoNLLData(dev_file, true, false);
			}
			
			System.err.println("[Info] "+TargetSentimentGlobal.LABELS.length+" labels: "+ Arrays.toString(TargetSentimentGlobal.LABELS));
			
			List<NeuralNetworkCore> fvps = new ArrayList<NeuralNetworkCore>();
			if(NetworkConfig.USE_NEURAL_FEATURES){
//				gnp =  new GlobalNetworkParam(OptimizerFactory.getGradientDescentFactory());
				if (neuralType.equals("lstm")) {
					String optimizer = nnOptimizer;
					boolean bidirection = true;
					fvps.add(new BidirectionalLSTM(hiddenSize, bidirection, optimizer, 0.05, 5, 3, gpuId, embedding));
					//fvps.add(new BidirectionalLSTM(hiddenSize, bidirection, optimizer, 0.05, 5, 3, gpuId, embedding));
				} else if (neuralType.equals("continuous")) {
					fvps.add(new TSFeatureValueProvider(TargetSentimentGlobal.Word2Vec, TSFeatureValueProvider.LABELS_CONNINUOUS.length).setUNK(UNK).setModelFile(model_file + ".nn"));
				} else {
					throw new RuntimeException("Unknown neural type: " + neuralType);
				}
			} 
			GlobalNeuralNetworkParam nn = new GlobalNeuralNetworkParam(fvps);
			GlobalNetworkParam gnp = new GlobalNetworkParam(optimizer, nn);
			
			
			
			Class<? extends VisualizationViewerEngine> visualizerClass = getViewer(modelname);
			
			
			model = createNetworkModel(modelname, gnp, neuralType);
			//NetworkModel model = DiscriminativeNetworkModel.create(fa, compiler);
			
			
			
			TargetSentimentGlobal.clearTemporalData();
			
			if (!SKIP_TRAIN)
			{
				model.train(trainInstances, num_iter);			
				
				if (!NetworkConfig.USE_NEURAL_FEATURES) {
				saveModel(model, gnp, model_file);
				}
			}
			if (visual) model.visualize(visualizerClass, trainInstances);
			
			if (!SKIP_TEST)
			{
				if (SKIP_TRAIN)
				{
					model = loadModel(model_file);
				}
				TargetSentimentGlobal.clearTemporalData();
				
				
				Instance[] predictions = model.decode(testInstances);
				
				//if (visual) model.visualize(visualizerClass, testInstances);
				
				writeResult(predictions, result_file);
			}
			
			nn.closeNNConnections();
		}
		return;
	}
	
	@SuppressWarnings("unchecked")
	private static TSInstance<Label>[] readCoNLLData(String fileName, boolean withLabels, boolean isLabeled) throws IOException{
		InputStreamReader isr = new InputStreamReader(new FileInputStream(fileName), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
		ArrayList<TSInstance<Label>> result = new ArrayList<TSInstance<Label>>();
		ArrayList<String[]> words = null;
		ArrayList<Label> labels = null;
		int numEntityinSentence = 0;
		int numDiscardInstance = 0;
		int numEntity = 0;
		int instanceId = 1;
		while(br.ready()){
			if(words == null){
				words = new ArrayList<String[]>();
			}
			if(withLabels && labels == null){
				labels = new ArrayList<Label>();
			}
			String line = br.readLine().trim();
			if(line.startsWith("##")){
				continue;
			}
			if(line.length() == 0){
				if(words.size() == 0){
					continue;
				}
				
				if (numEntityinSentence > 0 || isLabeled == false) {
					TSInstance<Label> instance = new TSInstance<Label>(instanceId, 1, words, labels);
					if (isLabeled) {
						instance.setLabeled(); // Important!
					} else {
						instance.setUnlabeled();
					}
					instanceId++;
					instance.preprocess();
					result.add(instance);
					numEntity += numEntityinSentence;
				} else {
					numDiscardInstance++;
				}
				words = null;
				labels = null;
				numEntityinSentence = 0;
			} else {
				int lastSpace = line.lastIndexOf(SEPERATOR);
				String[] features = line.substring(0, lastSpace).split(SEPERATOR);
				words.add(features);
				if(withLabels){
					String labelStr = line.substring(lastSpace+1);
					//labelStr = labelStr.replace("B-", "I-");
					Label label = TargetSentimentGlobal.getLabel(labelStr);
					labels.add(label);
					if (!labelStr.equals("O"))
						numEntityinSentence++;
				}
			}
		}
		br.close();
		System.out.println("There are " + numEntity + " entities in total.");
		System.out.println(numDiscardInstance + " instances are discarded.");
		return result.toArray(new TSInstance[result.size()]);
	}
	
	@SuppressWarnings("unchecked")
	private static TSInstance<Label>[] readAdditionalCoNLLData(TSInstance<Label>[] mainInstances, String fileName, boolean withLabels, boolean isLabeled) throws IOException{
		
		TSInstance<Label> sample = mainInstances[0];
		int featureNum = sample.getInput().get(0).length;
		
		InputStreamReader isr = new InputStreamReader(new FileInputStream(fileName), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
		ArrayList<TSInstance<Label>> result = new ArrayList<TSInstance<Label>>();
		
		for(int i = 0; i < mainInstances.length; i++)
		{
			result.add(mainInstances[i]);
		}
		
		/*{
			result.clear();
			result.add(mainInstances[0]);
		}*/
		int mainDataSize = result.size();
		
		ArrayList<String[]> words = null;
		ArrayList<Label> labels = null;
		int instanceId = mainDataSize + 1;
		int num_O = 0;
		while(br.ready()){
			if(words == null){
				words = new ArrayList<String[]>();
			}
			if(withLabels && labels == null){
				labels = new ArrayList<Label>();
			}
			String line = br.readLine().trim();
			if(line.startsWith("##")){
				continue;
			}
			if(line.length() == 0){
				if(words.size() == 0 || num_O == labels.size()){
					continue;
				}
				TSInstance<Label> instance = new TSInstance<Label>(instanceId, 1, words, labels);
				instance.setAdditional(true);
				if(isLabeled){
					instance.setLabeled(); // Important!
				} else {
					instance.setUnlabeled();
				}
				instanceId++;
				result.add(instance);
				words = null;
				labels = null;
				num_O = 0;
			} else {
				int lastSpace = line.lastIndexOf(SEPERATOR);
				String[] features_split = line.substring(0, lastSpace).split(SEPERATOR);
				String[] features = new String[featureNum];
				
				//features[0] = features_split[0];
				
				for(int i = 0; i < features.length; i++)
				{
					if (i < features_split.length)
						features[i] =  features_split[i];
					else
						features[i] = "_";
				}
				
				words.add(features);
				if(withLabels){
					String labelStr = line.substring(lastSpace+1);
					if (!labelStr.equals("O"))
					{
						
						String suffix = labelStr.substring(2);
						if (TargetSentimentGlobal.NETHashSet.contains(suffix))
						{
							labelStr = labelStr.substring(0, 2) + "unknown";
						}
						else
						{
							labelStr = "O";
						}
					}
					else
						num_O++;
					//labelStr = labelStr.replace("B-", "I-");
					Label label = TargetSentimentGlobal.getLabel(labelStr);
					labels.add(label);
					
					
				}
			}
		}
		br.close();
		
		System.out.println("Additional Instance:" + (result.size() - mainDataSize));
		
		return result.toArray(new TSInstance[result.size()]);
	}
	
	
	@SuppressWarnings("unchecked")
	private static TSInstance<Label>[] readAdditionalSentimentData(TSInstance<Label>[] mainInstances, String fileName, boolean withLabels, boolean isLabeled) throws IOException{
		
		TSInstance<Label> sample = mainInstances[0];
		int featureNum = sample.getInput().get(0).length;
		
		InputStreamReader isr = new InputStreamReader(new FileInputStream(fileName), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
		ArrayList<TSInstance<Label>> result = new ArrayList<TSInstance<Label>>();
		
		for(int i = 0; i < mainInstances.length; i++)
		{
			result.add(mainInstances[i]);
		}
		
		
		int mainDataSize = result.size();
		
		ArrayList<String[]> words = null;
		ArrayList<Label> labels = null;
		int instanceId = mainDataSize + 1;

		while(br.ready()){
			if(words == null){
				words = new ArrayList<String[]>();
			}
			if(withLabels && labels == null){
				labels = new ArrayList<Label>();
			}
			String line = br.readLine().trim();
			if(line.startsWith("##")){
				continue;
			}
			if(line.length() == 0){
				if(words.size() == 0){
					break;
				}
				
			} else {
				int lastSpace = line.lastIndexOf(SEPERATOR);
				String[] features_split = line.split(SEPERATOR);
				String[] words_arr = features_split[2].split(" ");
				for(int i = 0; i < words_arr.length; i++)
				{
					words.add(new String[]{words_arr[i]});
				}
				
				
				
				if(withLabels){
					String labelStr = features_split[1];
					int id = Integer.parseInt(labelStr);
					
					if (id == 0)
					{
						labelStr = "neutral";
						id = 0;
					}
					else if (id > 0)
					{
						labelStr = "positive";
						id = 1;
					}
					else if (id < 0)
					{
						labelStr = "negative";
						id = -1;
					}
					
					Label label = new Label(labelStr, id);
					labels.add(label);
					
				}
				
				TSInstance<Label> instance = new TSInstance<Label>(instanceId, 1, words, labels);
				instance.setAdditional(true);
				instance.setSentenceLevelInstance();
				if(isLabeled){
					instance.setLabeled(); // Important!
				} else {
					instance.setUnlabeled();
				}
				instanceId++;
				result.add(instance);
				words = null;
				labels = null;
			}
		}
		br.close();
		
		System.out.println("Additional Instance:" + (result.size() - mainDataSize));
		
		return result.toArray(new TSInstance[result.size()]);
	}
	
	private static void writeResult(Instance[] pred, String filename_output)
	{
		//String filename_output = (String) getParameters("filename_output");
		//String filename_standard =  (String) getParameters("filename_standard");
		
		PrintWriter p = null;
		try {
			p = new PrintWriter(new File(filename_output), "UTF-8");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		if (DEBUG)
			System.out.println("POS Tagging Result: ");
		for(int i = 0; i < pred.length; i++)
		{
			if (DEBUG)
				System.out.println("Testing case #" + i + ":");
			
			
			
			ArrayList<String[]> input = (ArrayList<String[]>)pred[i].getInput();
			ArrayList<Label> output = (ArrayList<Label>)pred[i].getPrediction();
			
			
			
			if (DEBUG)
			{
				System.out.println(input);
				System.out.println(output);
			}
			for(int j = 0; j < input.size(); j++)
			{
				//try{
				p.write(output.get(j).getForm() + "\n");
				//} catch (Exception e) {
				//	System.err.println();
				//}
			}
			
			p.write("\n");
		}
		
		p.close();
		
		if (DEBUG)
		{
			System.out.println("\n");
		}
		System.out.println(modelname + " Evaluation Completed");
		
		if (TargetSentimentGlobal.OUTPUT_SENTIMENT_SPAN)
		{
			try {
				p = new PrintWriter(new File(filename_output + ".span.html"), "UTF-8");
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (UnsupportedEncodingException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
			String css = "/Users/Leo/workspace/ui/overlap.css";
			if (NetworkConfig.OS.equals("linux")) {
				css = "/home/lihao/workspace/ui/overlap.css";
			}
			
			String header = "<html><head><link rel='stylesheet' type='text/css' href='" + css + "' /></head> <body><br><br>\n";
			String footer = "\n</body></html>";
			p.write(header);
			
			int pInst = 0, counter = 0;
			int[][] splits = new int[pred.length][];
			ArrayList<Integer> split = new ArrayList<Integer>();
			
			
			for(int i = 0; i < pred.length; i++)
			{
				TSInstance inst = (TSInstance)pred[i];
				ArrayList<String[]> input = (ArrayList<String[]>)inst.getInput();
				ArrayList<Label> output = (ArrayList<Label>)inst.getPrediction();
				ArrayList<int[]> scopes = inst.scopes;
						
				int pSplit = 0;
				String t = "";
				
				char lastTag = 'O';
				
				String scopeText = "";
				int entityIdx = 0;
				
				for(int k = 0; k < input.size(); k++) {
					
					String labelStr = output.get(k).getForm();
					char tag = labelStr.charAt(0);
					
					if (lastTag != 'O' && tag != 'I') {
						t += "<span class='tooltiptext'>" + scopeText + "</span></div>  ";
						scopeText = "";
					}
					
					
					if (tag == 'B')
					{
						t += "<div class='tooltip entity_" + labelStr.substring(2) + "'>";
						scopeText = "";
						for(int j = scopes.get(entityIdx)[0]; j <  scopes.get(entityIdx)[1]; j++) {
							scopeText += input.get(j)[0] + " ";
						}
						
						entityIdx++;
					}
					
					t += input.get(k)[0] + " ";
					
					
					lastTag = tag;
				}
				
				if (lastTag != 'O') {
					t += "<span class='tooltiptext'>" + scopeText + "</span></div>  ";;
				}
				
				t += "<br>";
				p.println(t);
				
				
				
				/*
				t = "";
				for(int j = 0; j < input.size(); j++)
				{
					t += output.get(j).getForm() + " ";
				}
				p.println(t);
				
				t = "";
				for(int k = 0; k < scopes.size(); k++) {
					int[] scope = scopes.get(k);
					
					t += scope[0] + "," + scope[1] + " ";
				}
				p.println(t);
				p.println();*/
			
			}
			
			p.write(footer);
			
			p.close();
					
			
		}
	
		
	}
	
	
	
	public static void processArgs(String[] args){
		
		if (args.length == 0)
		{
			return;
		}
		
		if(args[0].equals("-h") || args[0].equals("help") || args[0].equals("-help") ){
			System.err.println("Sentiment Scope Version: Joint Entity Recognition and Sentiment Prediction TASK: ");
			//System.err.println("\t usage: java -jar dpe.jar -trainNum -1 -testNum -1 -thread 5 -iter 100 -pipe true");
			//System.err.println("\t put numTrainInsts/numTestInsts = -1 if you want to use all the training/testing instances");
			System.exit(0);
		}else{
			for(int i=0;i<args.length;i=i+2){
				switch(args[i]){
					case "-modelname": modelname = args[i+1]; break;   //default: all 
					case "-reg": l2 = Double.valueOf(args[i+1]);  break;
					case "-num_iter": num_iter = Integer.valueOf(args[i+1]); break;    //default:all
					case "-beginindex": begin_index = Integer.valueOf(args[i+1]); break;    //default:all
					case "-endindex": end_index = Integer.valueOf(args[i+1]); break;   //default:100;
					case "-lang": lang = args[i+1]; break;
					case "-mentionpenalty" : mentionpenalty = Boolean.getBoolean(args[i+1]); break;
					case "-subpath" : subpath = args[i+1]; break;
					case "-NEMaxLength": NEMaxLength = Integer.valueOf(args[i+1]); break;
					case "-thread": numThreads = Integer.valueOf(args[i+1]); break;   //default:5
					case "-emb" : embedding = args[i+1]; break;
					case "-gpuid": gpuId = Integer.valueOf(args[i+1]); break;
					case "-usepostag" : TargetSentimentGlobal.USE_POS_TAG = Boolean.parseBoolean(args[i+1]); break;
					case "-useadditional" : 
						additionalDataset = args[i+1];
						if (additionalDataset.equals("none")) break;
						TargetSentimentGlobal.ENABLE_ADDITIONAL_DATA = true; 
						break;
					case "-fixne" : TargetSentimentGlobal.FIXNE = Boolean.parseBoolean(args[i+1]); break;
					case "-dumpfeature" : TargetSentimentGlobal.DUMP_FEATURE = Boolean.parseBoolean(args[i+1]); break;
					case "-visual" : visual = Boolean.parseBoolean(args[i+1]); break;
					case "-skiptest" : SKIP_TEST = Boolean.parseBoolean(args[i+1]); break;
					case "-skiptrain" : SKIP_TRAIN = Boolean.parseBoolean(args[i+1]); break;
					//case "-windows":ECRFEval.windows = true; break;            //default: false (is using windows system to run the evaluation script)
					//case "-batch": NetworkConfig.USE_BATCH_TRAINING = true;
					//				batchSize = Integer.valueOf(args[i+1]); break;
					//case "-model": NetworkConfig.MODEL_TYPE = args[i+1].equals("crf")? ModelType.CRF:ModelType.SSVM;   break;
					case "-neural": if(args[i+1].equals("mlp") || args[i+1].equals("lstm")|| args[i+1].equals("continuous")){ 
											NetworkConfig.USE_NEURAL_FEATURES = true;
											neuralType = args[i+1]; //by default optim_neural is false.
											NetworkConfig.IS_INDEXED_NEURAL_FEATURES = false; //only used when using the senna embedding.
											NetworkConfig.REGULARIZE_NEURAL_FEATURES = true;
									}
									break;
					case "-initNNweight": 
						NetworkConfig.INIT_FV_WEIGHTS = args[i+1].equals("true") ? true : false; //optimize the neural features or not
						break;
					case "-optimNeural": 
						NetworkConfig.OPTIMIZE_NEURAL = args[i+1].equals("true") ? true : false; //optimize the neural features or not
						if (!NetworkConfig.OPTIMIZE_NEURAL) {
							nnOptimizer = args[i+2];
							i++;
						}break;
					case "-optimizer":
						 if(args[i+1].equals("sgd")) {
							 optimizer = OptimizerFactory.getGradientDescentFactoryUsingGradientClipping(0.05, 5);
							 
						 }
						break;
					
					
					
					//case "-lr": adagrad_learningRate = Double.valueOf(args[i+1]); break;
					case "-backend": NetworkConfig.NEURAL_BACKEND = args[i+1]; break;
					case "-os": NetworkConfig.OS = args[i+1]; break; // for Lua native lib, "osx" or "linux" 
					case "-embsuffix": embedding_suffix = args[i+1]; break;
					case "-unk" : UNK = args[i+1]; System.out.println("set UNK = " + UNK);break;
					case "-ngram" : TargetSentimentGlobal.NGRAM = Boolean.parseBoolean(args[i+1]); break;
					case "-hidesink" : VisualizeGraph.hideEdgetoSink = Boolean.parseBoolean(args[i+1]); break;
					case "-overlap":TargetSentimentGlobal.OVERLAPPING_FEATURES = Boolean.parseBoolean(args[i+1]); break;
					case "-outputscope": TargetSentimentGlobal.OUTPUT_SENTIMENT_SPAN = Boolean.parseBoolean(args[i+1]); break;
					case "-hiddensize": hiddenSize = Integer.valueOf(args[i+1]); break;    //default:all
					case "-NERL": TargetSentimentGlobal.NER_SPAN_MAX = Integer.valueOf(args[i+1]); break;  
					
					default: System.err.println("Invalid arguments "+args[i]+", please check usage."); System.exit(0);
				}
			}
			System.err.println("[Info] beginIndex: "+begin_index);
			System.err.println("[Info] endIndex: "+end_index);
			System.err.println("[Info] numIter: "+ num_iter);
			System.err.println("[Info] numThreads: "+numThreads);
			System.err.println("[Info] Regularization Parameter: "+ l2);
		}
	}
	
	public static NetworkModel createNetworkModel(String modelname, GlobalNetworkParam gnp, String neuralType) {
		String modelpath = "org.statnlp.targetedsentiment.f";
		FeatureManager fm = null;
		NetworkCompiler compiler = null;


		if (modelname.equals("sentimentspan_latent")) {
			fm = new org.statnlp.targetedsentiment.f.latent.TargetSentimentFeatureManager(gnp, neuralType, false);
			compiler = new org.statnlp.targetedsentiment.f.latent.TargetSentimentCompiler();
		} else if (modelname.equals("baseline_collapse")) {
			fm = new org.statnlp.targetedsentiment.f.baseline.CollapseTSFeatureManager(gnp, neuralType, false);
			compiler = new org.statnlp.targetedsentiment.f.baseline.CollapseTSCompiler();
		} else if (modelname.equals("baseline_collapse_simple")) {
			fm = new org.statnlp.targetedsentiment.f.baseline.simple.CollapseSimpleTSFeatureManager(gnp, neuralType, false);
			compiler = new org.statnlp.targetedsentiment.f.baseline.CollapseTSCompiler();
		} else if (modelname.equals("sentimentscope_overlap")) {
			fm = new org.statnlp.targetedsentiment.overlap.cont.SSOverlapContFeatureManager(gnp, neuralType, false);
			compiler = new org.statnlp.targetedsentiment.overlap.cont.SSOverlapContCompiler();
		}else if (modelname.equals("sentiment_parsing")) {
			fm = new org.statnlp.targetedsentiment.ncrf.semi.SentimentParsingSemiFeatureManager(gnp, neuralType, false);
			compiler = new org.statnlp.targetedsentiment.ncrf.semi.SentimentParsingSemiCompiler();
		}else if (modelname.equals("sentiment_parsing_hybrid")) {
			fm = new org.statnlp.targetedsentiment.ncrf.semi.SentimentParsingSemiFeatureManager(gnp, neuralType, false);
			compiler = new org.statnlp.targetedsentiment.ncrf.semi.SentimentParsingSemiHybridCompiler();
		}

		NetworkModel model = DiscriminativeNetworkModel.create(fm, compiler);

		return model;
	}
	
	public static Class<? extends VisualizationViewerEngine> getViewer(String modelname)
	{
		String visualModelPath = null;
		switch (modelname)
		{
		case "sentimentspan_latent": return null; 
		case "baseline_collapse":visualModelPath = "f.baseline.CollapseViewer";break;
		case "sentimentscope_overlap":visualModelPath = "overlap.cont.SSOverlapContViewer"; break;
		case "sentiment_parsing":visualModelPath = "ncrf.semi.SentimentParsingSemiViewer"; break;
		case "sentiment_parsing_hybrid":visualModelPath = "ncrf.semi.SentimentParsingSemiViewer"; break;
		}
		
		if (visualModelPath == null)
			return null;
		
		String visualizerModelName = "org.statnlp.targetedsentiment." + visualModelPath;
		Class<? extends VisualizationViewerEngine> visualizerClass = null;
		
		try {
			visualizerClass = (Class<VisualizationViewerEngine>) Class.forName(visualizerModelName);
		} catch (ClassNotFoundException e) {
			System.err.println("Class not found");
		}
		
		return visualizerClass;
	}
	
	public static void saveModel(NetworkModel model, GlobalNetworkParam param, String filename_model ) throws IOException {
		
		
		System.out.println();
		System.err.println("Saving Model...");
		ObjectOutputStream out;
		out = new ObjectOutputStream(new FileOutputStream(filename_model));
		
		out.writeObject(model);
		out.flush();
		out.close();
		System.err.println("Model Saved.");
		
		if (TargetSentimentGlobal.DUMP_FEATURE)
			printFeature(param, filename_model);
		
	}
	
	public static void printFeature(GlobalNetworkParam param, String filename_model )
	{
		
		StringIndex string2Idx = param.getStringIndex();
		string2Idx.buildReverseIndex();
		
		PrintWriter modelTextWriter = null;
		try {
			modelTextWriter = new PrintWriter(filename_model + ".dump");
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}


		modelTextWriter.println("Num features: "+param.countFeatures());

		modelTextWriter.println("Features:");

		TIntObjectHashMap<TIntObjectHashMap<TIntIntHashMap>> featureIntMap =  param.getFeatureIntMap();

		for(int featureTypeId: sorted(featureIntMap.keys())){ //sorted

		     //.println(featureType);

			 TIntObjectHashMap<TIntIntHashMap> outputInputMap = featureIntMap.get(featureTypeId);

		     for(int outputId: sorted(outputInputMap.keys())){ //sorted

		          //modelTextWriter.println("\t"+output);

		    	 TIntIntHashMap inputMap = outputInputMap.get(outputId);

		          for(int inputId: inputMap.keys()){

		               int featureId = inputMap.get(inputId);
		               
		               String featureType = string2Idx.get(featureTypeId);
		               String input = string2Idx.get(inputId);
		               String output = string2Idx.get(outputId);

		               modelTextWriter.println(featureType + input+ ":= " + output + "="+param.getWeight(featureId));
		               if (SpanModelGlobal.ECHO_FEATURE)
		            	   System.out.println(featureType + input+ ":= " + output + "="+param.getWeight(featureId));
		          
		          }

		     }
		     
		     modelTextWriter.flush();

		}

		modelTextWriter.close();
	}
	
	static int[] sorted(int[] arr)
	{
		int[] arr_sorted = arr.clone();
		Arrays.sort(arr_sorted);
		return arr_sorted;
	}

	public static NetworkModel loadModel(String filename_model) throws IOException {
		
		
		NetworkModel model = null;
		System.err.println("Loading Model...");
		ObjectInputStream in = new ObjectInputStream(new FileInputStream(filename_model));
		try {
			model = (NetworkModel)in.readObject();
		} catch (ClassNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		in.close();
			
		
		System.err.println("Model Loaded.");

		return model;		
	}
}
