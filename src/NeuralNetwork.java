import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;

/**
 * 
 */

/**
 * @author Dion
 *
 */
public class NeuralNetwork {

	// hard coded size of the MNIST images
	private static final int IMAGE_WIDTH = 28;
	private static final int IMAGE_HEIGHT = 28;
	
	// the number of neurons on our input layer, 1 per greyscale pixel input
	private static final int INPUT_NEURONS = IMAGE_WIDTH * IMAGE_HEIGHT;
	
	// number of possible output options the network can identify [10 = 0-9 digits]
	private static final int OUTPUT_NEURONS = 10;
	
	// size of the hidden layer of neurons, confined to 1 layer here
 	private static final int HIDDEN1_NEURONS = 10;
	
	// our input layer of neurons
	private Neuron[] inputLayer = null;
	private Neuron[] hiddenLayer = null;
	private Neuron[] outputLayer = null;
	
	// our learning rate for the network
	private double learnN;

	// our activation function
	private static double sigmoid(double z) {
		
		return 1.0 / (1.0 + Math.exp(-z));	
	}

	// sets up the weights of a layer based on it's attachment.  part of initializing the network
	private void attachLayers(Neuron[] layerA, Neuron[] layerB) {
		
		for(Neuron n:layerB) {
			n.setNumWeights(layerA.length);
		}
	}
	
	// propagate the values from one layer to the next
	private void forwardPropagate(Neuron[] layerA, Neuron[] layerB) {
		
		for(Neuron n:layerB) {
			double sumReg = 0.0;
			for(int i = 0; i < n.getNumWeights(); i++) {
				sumReg += n.getWeight(i) * layerA[i].getOutput();
			}
			
			// consider moving the sigmoid function into the Neuron??
			n.setOutput(sigmoid(sumReg + n.getBias()));
		}
	}

	// move the entire network ahead with propagation
	private void forwardPropagate() {

		// assumes the input layer is setup here with an image...
		forwardPropagate(inputLayer, hiddenLayer);
		forwardPropagate(hiddenLayer, outputLayer);
	}

	// second/third? attempt at backPropagation...
	// 'vT' represents the ideal vector that the output neurons should register
	// it is created by turning the file label into a 10-dimensional vector of binary values [0.0,1.0]
	// 'debug' turns on/off console logging (performance)
	private void backPropagate2(double[] vT, boolean debug) {
	
		// let's calculate total error from the output layers and dump to console
		if(debug) {
			double Etotal = 0.0;
			for(int i = 0; i < vT.length; i++) {
				double mse = outputLayer[i].getOutput() - vT[i];
				Etotal += (mse * mse);
			}
			Etotal *= 0.5;
			
			System.out.format("Total Error: %1$7.4f %n", (float)Etotal);
		}
		
		// let's start by getting the error relative to each output node
		// right now the error value is temporarily housed in the neuron, but for optimization
		// purposes, we can probably make a more transient store for this later
		for(int i = 0; i < outputLayer.length; i++) {
			
			Neuron n = outputLayer[i];
			
			// the error with respect to the activated output
			// the error with respect to the net output
			double dEdA = -(vT[i] - n.getOutput());
			double dEdN = n.getOutput() * (1.0 - n.getOutput());

			// total error on the net side of this neuron
			n.setError(dEdA * dEdN);
		}
		
		
		// now let's get the errors on the hidden layer added up
		for(int i = 0; i < hiddenLayer.length; i++) {
			
			Neuron n = hiddenLayer[i];
					
			// the error on this neuron is the sum of all the errors on the outputs relative to their connections	
			// convert to net error before we apply
			double dE = 0.0;
			for(Neuron m:outputLayer) { 
				dE += m.getError() * m.getWeight(i);
			}
			dE *= n.getOutput() * (1.0 - n.getOutput());

			n.setError(dE);
		}
		
		// adjust the weights between input and hidden layer
		for(int i = 0; i < hiddenLayer.length; i++) {
		
			Neuron n = hiddenLayer[i];
			for(int j = 0; j < n.getNumWeights(); j++) {
			
				double dEdW = n.getError() * inputLayer[j].getOutput();
				double wP = n.getWeight(j) - (learnN * dEdW);
				n.setWeight(j, wP);
			}		
		}
		
		// adjust the weights between hidden and output layers
		for(int i = 0; i < outputLayer.length; i++) {
			
			Neuron n = outputLayer[i];
			for(int j = 0; j < n.getNumWeights(); j++) {
			
				double dEdW = n.getError() * hiddenLayer[j].getOutput();
				double wP = n.getWeight(j) - (learnN * dEdW);
				n.setWeight(j, wP);
			}		
		}
		
		// back propagation is complete
	}
	
	// called as part of setup to randomize the initial state of the network
	private void randomizeNetwork(double N) {
		
		// skip the input layer randomization as it's not initialized or needed
		
		for(Neuron n:hiddenLayer) {
			n.randomize(N);
		}
		for(Neuron n:outputLayer) {
			n.randomize(N);
		}
	}

	// attach the image to the Input Neurons to prepare for training or testing
	private void attachImage(byte[] mnist) {
		
		for(int i = 0; i < mnist.length; i++) {
			int a = mnist[i] & 0xff;
			inputLayer[i].setOutput((double)a / 255.0);
		}
	}

	// create a vector from the output values of the Output Neurons
	public double[] vectorFromOutput() {
		
		double[] vector = new double[outputLayer.length];
		for(int i = 0; i < vector.length; i++) {
			vector[i] = outputLayer[i].getOutput();
		}
		return vector;
	}

	// given a label (0-N), initialize a target vector to match (e.g. 3 = [0, 0, 0, 1.0, 0.0])
	private double[] vectorFromLabel(byte label) {

		double[] vector = new double[outputLayer.length];	
		for(int i = 0; i < vector.length; i++) {
			if(i == (int)label) 
				vector[i] = 1.0;
			else				
				vector[i] = 0.0;
		}
		
		return vector;		
	}

	// return the label of the network's output as an integer from 0-9 (ideally!)
	public int labelFromVector(double[] vector) {
		
		int index = -1;  		// means not found
		double bigValue = -1000.0;	// should be small enough, right?
		
		for(int i = 0; i < vector.length; i++) {
			if(vector[i] > bigValue) {
				index = i;
				bigValue = vector[i];
			}
		}
		
		return index;		
	}
	
	/**
	 * 
	 */
	public NeuralNetwork() {
		
		learnN = 1.0 / Math.sqrt(HIDDEN1_NEURONS);
	}
	
	// call with a learning rate specified.  If default constuctor called, 
	// learning rate is 0.25
	public NeuralNetwork(double N) {
		learnN = N;
	}
	
	public void initNetwork() {

		// the input layer is designed to handle 28 x 28 images
		// the input layer has no weights assigned
		inputLayer = new Neuron[INPUT_NEURONS];
		for(int i = 0; i < INPUT_NEURONS; i++)
			inputLayer[i] = new Neuron();
		
		// our hidden layer is shallow
		hiddenLayer = new Neuron[HIDDEN1_NEURONS];
		for(int i = 0; i < HIDDEN1_NEURONS; i++)
			hiddenLayer[i] = new Neuron();
		
		// and our final output layer handles our 10 digits 0-9
		outputLayer = new Neuron[OUTPUT_NEURONS];
		for(int i = 0; i < OUTPUT_NEURONS; i++)
			outputLayer[i] = new Neuron();

		// hook up the layers.  this really just initializes the weights appropriately
		attachLayers(inputLayer, hiddenLayer);
		attachLayers(hiddenLayer, outputLayer);

		// randomize the initial weights and biases in our network
		randomizeNetwork((double)(inputLayer.length + hiddenLayer.length));
	
		
		// OKAY!  Neural Network is setup and ready for action.  Time to start feeding it images
		// and iterating towards a result!
		
	}
	
	// teach the network by sending it an image and label
	public void trainNetwork(byte[] mnist, byte label) {

		// setup our target vector 
		double[] targetVector = vectorFromLabel(label);

		// setup the image in the input layer
		attachImage(mnist);
		
		// forward propagate these inputs throughout the network
		forwardPropagate();
		
		// calculate the error and apply backwards to the weights and biases
		backPropagate2(targetVector, false);
	}
	
	// test the network by sending an image and getting a response
	public int testNetwork(byte[] mnist) {

		// get the image setup in the input layer
		attachImage(mnist);
		
		// forward propagate this image
		forwardPropagate();
		
		// let's make our guess from our Output Neurons output values
		return(labelFromVector(vectorFromOutput()));
	}
	
	// reverse activate a network to generate an image of it's 'memory'
	public void reverseActivateImage(String file, int output) {

		// let's set the right values on the output neurons
		for(int i = 0; i < outputLayer.length; i++) {
			
			if(i == output) {
				outputLayer[i].setOutput(1.0);
			}
			else {
				outputLayer[i].setOutput(0.0);
			}
		}
		
		// okay, now let's back propagate this signal based on connections
		for(int i = 0; i < hiddenLayer.length; i++) {
			
			double hOutput = 0.0;
			for(int j = 0; j < outputLayer.length; j++) {
				hOutput += outputLayer[j].getOutput() * outputLayer[j].getWeight(i);
			}
			hiddenLayer[i].setOutput(hOutput);
		}
		
		// now, let's set the input signals based on these relationships
		for(int i = 0; i < inputLayer.length; i++) {
			
			double iOutput = 0.0;
			for(int j = 0; j < hiddenLayer.length; j++) {
				iOutput += hiddenLayer[j].getOutput() * hiddenLayer[j].getWeight(i);
			}
			inputLayer[i].setOutput(iOutput);
		}
		
		// Create a new buffered image
		BufferedImage bi = new BufferedImage(28, 28, BufferedImage.TYPE_INT_ARGB);
		for(int y = 0; y < 28; y++) {
			for(int x = 0; x < 28; x++) {
				int color = (int)(inputLayer[y*28+x].getOutput() * 255.0);
				if(color < 0) color = 0;
				if(color > 255) color = 255;
				Color col = new Color(color, color, color);
				bi.setRGB(x, y, col.getRGB());
			}
		}
		
		try {
			File outputFile = new File(file);
			ImageIO.write(bi, "png", outputFile);
		}
		catch(IOException e) {
		}
	}
	
	// dump a status of the network
	public void printNetwork() {
		
		System.out.println("Neural Network Dump");
		
		// describe the input layer
		System.out.println("Input Layer: " + inputLayer.length + " nodes");
		System.out.println("OUTPUTS");
		for(int i = 0; i < 28; i++) {
			
			System.out.print("[");
			for(int j = 0; j < 28; j++) {
				System.out.format("%1$3.2f", (float)inputLayer[i * 28 + j].getOutput());
				if(j < 27)
					System.out.print(", ");
			}
			System.out.println("]");
		}
		
		// describe the hidden layer
		System.out.println("Hidden Layer: " + hiddenLayer.length + " nodes");
		System.out.println("OUTPUTS");
		System.out.print("[");
		for(int i = 0; i < hiddenLayer.length; i++) {
			System.out.format("%1$5.3f", (float)hiddenLayer[i].getOutput());
			if(i < hiddenLayer.length - 1)
				System.out.print(", ");
		}
		System.out.println("]");
		
		System.out.println("ERRORS");
		System.out.print("[");
		for(int i = 0; i < hiddenLayer.length; i++) {
			System.out.format("%1$5.3f", (float)hiddenLayer[i].getError());
			if(i < hiddenLayer.length - 1)
				System.out.print(", ");
		}
		System.out.println("]");
		
		// describe the output layer
		System.out.println("Output Layer: " + outputLayer.length + " nodes");
		System.out.println("OUTPUTS");
		System.out.print("[");
		for(int i = 0; i < outputLayer.length; i++) {
			System.out.format("%1$5.3f", (float)outputLayer[i].getOutput());
			if(i < outputLayer.length - 1)
				System.out.print(", ");
		}
		System.out.println("]");
		
		System.out.println("ERRORS");
		System.out.print("[");
		for(int i = 0; i < outputLayer.length; i++) {
			System.out.format("%1$5.3f", (float)outputLayer[i].getError());
			if(i < outputLayer.length - 1)
				System.out.print(", ");
		}
		System.out.println("]");
	}
}
