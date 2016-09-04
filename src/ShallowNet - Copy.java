import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

/**
 * 
 */

/**
 * @author Dion
 *
 */
public class ShallowNet {

	// the files used to train and test the network: MNIST sample data
	private String trainingImages = "MNIST/train-images.idx3-ubyte";
	private String trainingLabels = "MNIST/train-labels.idx1-ubyte";
	private String testingImages = "MNIST/test-images-idx3-ubyte";
	private String testingLabels = "MNIST/test-labels-idx1-ubyte";
	
	// our (shallow) neural network object
	private NeuralNetwork m_nn = null;
	
	// the neuron metadata and storage
	private int[] layerSize = {28 * 28, 20, 10};
	private double[][] weight;
	private double[][] bias;

	
	// Creates the memory for the network and sets initial values
	private void initializeNeurons() {
	
		// okay, weights and biases...
		// an n-layered system has n-1 columns of these
		// each neuron is connected to m neurons from the layer before
		// each neuron has 1 bias per
		weight = new double[layerSize.length - 1][];
		bias = new double[layerSize.length - 1][];
		
		for(int i = 0; i < layerSize.length - 1; i++) {
			weight[i] = new double[layerSize[i] * layerSize[i+1]];
			bias[i] = new double[layerSize[i+1]];
		}
		
		// now, let's initialize these guys with default values (random)
		for(int i = 0; i < layerSize.length - 1; i++) {
			
			for(int j = 0; j < weight[i].length; j++) {
				weight[i][j] = Math.random();
			}
			
			for(int j = 0; j < bias[i].length; j++) {
				bias[i][j] = Math.random();
			}
		}
	}
	
	private void trainNetwork() {

		FileInputStream fisImage = null;
		FileInputStream fisLabel = null;
		
		try {
			
			fisImage = new FileInputStream(trainingImages);
			fisLabel = new FileInputStream(trainingLabels);
			
			// read header from the images file and print it out 
			int imagesMagic = (fisImage.read() << 24) | (fisImage.read() << 16) | (fisImage.read() << 8) | fisImage.read();
			int imagesCount = (fisImage.read() << 24) | (fisImage.read() << 16) | (fisImage.read() << 8) | fisImage.read();
			int imagesRows = (fisImage.read() << 24) | (fisImage.read() << 16) | (fisImage.read() << 8) | fisImage.read();
			int imagesCols = (fisImage.read() << 24) | (fisImage.read() << 16) | (fisImage.read() << 8) | fisImage.read();
			System.out.println(trainingImages);
			System.out.println("Magic: " + imagesMagic + ", Count: " + imagesCount + ", Size: " + imagesRows + " x " + imagesCols);
			
			// read header from the labels file and print it out
			int labelsMagic = (fisLabel.read() << 24) | (fisLabel.read() << 16) | (fisLabel.read() << 8) | fisLabel.read();
			int labelsCount = (fisLabel.read() << 24) | (fisLabel.read() << 16) | (fisLabel.read() << 8) | fisLabel.read();
			System.out.println(trainingLabels);
			System.out.println("Magic: " + labelsMagic + ", Count: " + labelsCount);

			// basic check: images and labels match?
			if(imagesCount != labelsCount)
				throw new IOException("ERROR: Image and label files don't have matching record counts");
			
			
			// start iterating through the images and corresponding labels
			byte[] imageData = new byte[imagesRows * imagesCols];
			byte label = 0;
			for(int i = 0; i < imagesCount; i++) {
				
				// read in the data of the 28 x 28 image (greyscale bytes)
				fisImage.read(imageData);

				// read in the label of this image
				label = (byte) fisLabel.read();
				
				// now, set these up as the input neurons and train...
				// ??
				double[] outputNeurons = activateImage(imageData);
				int nnLabel = labelFromVector(outputNeurons);
				System.out.println("Image[" + (i+1) + "] Real: " + label + ", Guess: " + nnLabel); 
			}
		}
		catch(FileNotFoundException e) {
			System.out.println(e.getMessage());
		}
		catch(IOException e) {
			System.out.println(e.getMessage());			
		}
		finally {

			if(fisImage != null) {
				try { fisImage.close();	} 
				catch(IOException e) { }
			}
			
			if(fisLabel != null) {
				try { fisLabel.close(); }
				catch(IOException e) { }
			}
		}
	}
	
	private double[] activateImage(byte[] input) {
		
		// setup our input nodes as normalized doubles
		double[] inputVector = new double[input.length];
		for(int i = 0; i < input.length; i++) {
			inputVector[i] = (double) input[i] / 255.0;
		}
		
		double[] neuron = null;
		for(int i = 1; i < layerSize.length; i++) {
			neuron = new double[layerSize[i]];
			
			// assign value to neuron
			for(int n = 0; n < neuron.length; n++) {

				neuron[n] = 0.0;
				
				// sum this neuron
				for(int m = 0; m < layerSize[i-1]; m++) {
					neuron[n] += (inputVector[m] * weight[i-1][(n * layerSize[i-1]) + m]);
					neuron[n] -= bias[i-1][n]; 
				}
				neuron[n] = sigmoid(neuron[n]);
			}
			
			inputVector = neuron;
		}
		
		// let's send back our final numbers
		return neuron;
	}
	
	private void testNetwork() {
		
	}
	
	// Our Activation function, the sigmoidal 
	private double sigmoid(double z) {
		return 1.0 / (1.0 + Math.exp(-z));
	}
	
	// get a proper comparison output vector given a label
	private double[] vectorFromLabel(byte label) {
		double[] vector = new double[layerSize[layerSize.length-1]];
		
		for(int i = 0; i < vector.length; i++) {
			if((i == (int)label) && (i < vector.length)) {
				vector[i] = 1.0;
			}
			else {
				vector[i] = 0.0;
			}
		}
		
		return vector;
	}
	
	// get a label identifier given a vector of outputs
	private int labelFromVector(double[] vector) {
		
		int index = -1;  // means not found
		double bigValue = -1000.0;
		
		for(int i = 0; i < vector.length; i++) {
			if(vector[i] > bigValue) {
				index = i;
				bigValue = vector[i];
			}
		}
		
		return index;
	}
	
	// print a vector
	private void printVector(double[] vector) {
	
		System.out.print("[");
		for(int i = 0; i < vector.length; i++) {
			System.out.format("%1$3.2f", (float)vector[i]);
			if(i < vector.length-1)
				System.out.print(", ");
		}
		System.out.println("]");
	}
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		// TODO Auto-generated method stub
	
		// Let's get our neural net setup
		ShallowNet sn = new ShallowNet();
		sn.initializeNeurons();

		// let's train our new network
		System.out.println("Training ShallowNet...");
		sn.trainNetwork();
		
		// and test it out for accuracy
		System.out.println("Testing ShallowNet...");
		
		// wrap up the show, we're done
		System.out.println("Complete - quitting now");
	}
}
