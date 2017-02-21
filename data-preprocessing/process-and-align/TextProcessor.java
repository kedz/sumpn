import java.nio.file.Path;
import static java.util.stream.Collectors.*;
import java.util.stream.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.io.BufferedWriter;
import java.io.IOException;
import java.lang.ProcessBuilder;
import java.lang.Process;
import java.lang.InterruptedException;

import edu.stanford.nlp.pipeline.*;
import edu.stanford.nlp.util.CoreMap;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import java.util.*;

class TextProcessor {

    private StanfordCoreNLP pipeline;
    private Path outputPath;
    private BufferedWriter writer;
    private int totalProcessed = 0;
    private int totalLines;

    TextProcessor(Path outputPath, int totalLines) {

        Properties props = new Properties();
        props.setProperty("annotators", "tokenize, ssplit, pos, lemma, ner"); //, parse, dcoref");
        props.setProperty("ssplit.newlineIsSentenceBreak", "two");
        props.setProperty("tokenize.options", "asciiQuotes=true");
        props.setProperty("ner.useSUTime", "false");
        pipeline = new StanfordCoreNLP(props);
        this.outputPath = outputPath;
        writer = null;
        this.totalLines = totalLines;
    }

    public void close() {
        if (writer != null) {
            try {
            writer.close();
            } catch(IOException e) {
                System.out.println(e);
            }
        }
    }

    public void processLine(String line)  {

        try {
            if (writer == null) {
                writer = Files.newBufferedWriter(outputPath);
            }
        } catch(IOException e) {
            System.out.println(e);
            System.exit(1);
        }

        String[] elems = line.split("\t");
        

        String document = elems[0];
        String url = elems[1];
        String title = elems[2];
        String highlights_line = elems[3];
        String grafs_line = elems[4];

        String grafs = grafs_line.replace("__NL__", "\n").replace("__TAB__", "\t").replace("|", "\n\n");
        String highlights = highlights_line.replace("__NL__", "\n").replace("__TAB__", "\t").replace("|", "\n\n");

        String textInput = grafs + "\n\n****\n\n" + highlights;

        Annotation ann = new Annotation(textInput);
        pipeline.annotate(ann);

        List<CoreMap> sentences = ann.get(SentencesAnnotation.class);

        ArrayList<String> targetStrings = new ArrayList<String>();
        ArrayList<String> inputStrings = new ArrayList<String>();

        boolean inHighlights = false;

        for(CoreMap sentence: sentences) {

            List<CoreLabel> tokens = sentence.get(TokensAnnotation.class);
            ArrayList<String> tokenStrings = new ArrayList<String>(tokens.size());

           
            for (CoreLabel token: tokens) {
                String word = token.get(TextAnnotation.class).replace("/", "_");
                String pos = token.get(PartOfSpeechAnnotation.class);
                String ne = token.get(NamedEntityTagAnnotation.class);
                 
                tokenStrings.add(String.join("/", new String[]{word,pos,ne}));

            }


            if (tokens.size() == 1 && tokens.get(0).toString().equals("****-1")) {
                inHighlights = true;
                continue;
            }

            if (inHighlights) {
                targetStrings.add(String.join(" ", tokenStrings));
            } else { 
                inputStrings.add(String.join(" ", tokenStrings));
            }
            
        }
        
        String input = String.join("|", inputStrings);
        String target = String.join("|", targetStrings);

        String output = String.join("\t", new String[]{document, url, title, target, input});
        try {
            writer.write(output);
            writer.write("\n");
        } catch(IOException e) {
            System.out.println(e);
            System.exit(1);
        }

        totalProcessed += 1;

        float perComplete =  100 * (float) totalProcessed / (float) totalLines;
        System.out.print(String.format("Processing %10d / %10d (%10.4f", totalProcessed, totalLines, perComplete) + "%)\r");
        System.out.flush();
        //if (totalProcessed % 100 == 0) {
        //    System.out.print(".");
        //    System.out.flush();
        //} 

    };

    public static void main(String[] args) {

        Path inputFile = null;
        Path outputFile = null;
        int numLines = 1;

        for (int i=0; i < args.length - 1; ++i) {
            if (args[i].equals("--input")) {
                inputFile = Paths.get(args[i+1]);
            } else if (args[i].equals("--output")) {
                outputFile = Paths.get(args[i+1]);
            } else if (args[i].equals("--input-size")) {
                numLines = Integer.parseInt(args[i+1]);
            }
        };

        if (inputFile == null || outputFile == null) {
            System.out.println("usage: java TextProcessor --input INPUTPATH --output OUTPUTPATH");
            System.exit(1);
        }

        try {
            Path outputDir = outputFile.getParent();
            Files.createDirectories(outputDir);

        } catch (IOException e) {
            System.out.println(e);
            System.out.println("Could not create parent directory of output file. Bailing!");
            System.exit(1);
        }

        TextProcessor tp = new TextProcessor(outputFile, numLines);

        try {
            Stream<String> stream = Files.lines(inputFile);
            stream.forEach(tp::processLine);
        } catch (IOException e) {
            System.out.println(e);

        }

        tp.close();
    };

}
