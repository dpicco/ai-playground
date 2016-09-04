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
	private String testingImages = "MNIST/t10k-images.idx3-ubyte";
	private String testingLabels = "MNIST/t10k-labels.idx1-ubyte";
	
	// our (shallow) neural network object
	private NeuralNetwork m_nn = null;
	
	// number of training epochs
	private static int TRAINING_EPOCHS = 25;

	// let's initialize our network
	public void init() {
		
		// let's create and initialize our NeuralNetwork container
		m_nn = new NeuralNetwork(1.0);
		m_nn.initNetwork();
		
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
			
			// read header from the labels file and print it out
			int labelsMagic = (fisLabel.read() << 24) | (fisLabel.read() << 16) | (fisLabel.read() << 8) | fisLabel.read();
			int labelsCount = (fisLabel.read() << 24) | (fisLabel.read() << 16) | (fisLabel.read() << 8) | fisLabel.read();

			// basic check: images and labels match?
			if(imagesCount != labelsCount)
				throw new IOException("ERROR: Image and label files don't have matching record counts");
			
			int correct = 0;
			
			// start iterating through the images and corresponding labels
			byte[] imageData = new byte[imagesRows * imagesCols];
			for(int i = 0; i < imagesCount; i++) {
							
				// read in the data of the 28 x 28 image (greyscale bytes)
				fisImage.read(imageData);

				// read in the label of this image
				byte label = (byte)fisLabel.read();

				// now, set these up as the input neurons and train...
				m_nn.trainNetwork(imageData, label);

				// show status
				int guess = m_nn.labelFromVector(m_nn.vectorFromOutput());
				if(guess == label)
					correct++;				
				
				// dump a status of the network once in a while
				if(i % 3000 == 0) {
					System.out.println((100.0 * (double)i / 60000.0) + "%... ");
				}
			}
			System.out.println("Result: " + correct + "/60,000 (" + (100.0 * (double)correct / 60000.0) + "%)");
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
	
	
	private void testNetwork() {
		FileInputStream fisImage = null;
		FileInputStream fisLabel = null;
		
		try {
			
			fisImage = new FileInputStream(testingImages);
			fisLabel = new FileInputStream(testingLabels);
			
			// read header from the images file and print it out 
			int imagesMagic = (fisImage.read() << 24) | (fisImage.read() << 16) | (fisImage.read() << 8) | fisImage.read();
			int imagesCount = (fisImage.read() << 24) | (fisImage.read() << 16) | (fisImage.read() << 8) | fisImage.read();
			int imagesRows = (fisImage.read() << 24) | (fisImage.read() << 16) | (fisImage.read() << 8) | fisImage.read();
			int imagesCols = (fisImage.read() << 24) | (fisImage.read() << 16) | (fisImage.read() << 8) | fisImage.read();
			
			// read header from the labels file and print it out
			int labelsMagic = (fisLabel.read() << 24) | (fisLabel.read() << 16) | (fisLabel.read() << 8) | fisLabel.read();
			int labelsCount = (fisLabel.read() << 24) | (fisLabel.read() << 16) | (fisLabel.read() << 8) | fisLabel.read();

			// basic check: images and labels match?
			if(imagesCount != labelsCount)
				throw new IOException("ERROR: Image and label files don't have matching record counts");
			
			int correct = 0;
			
			// start iterating through the images and corresponding labels
			byte[] imageData = new byte[imagesRows * imagesCols];
			for(int i = 0; i < imagesCount; i++) {
							
				// read in the data of the 28 x 28 image (greyscale bytes)
				fisImage.read(imageData);

				// read in the label of this image
				byte label = (byte)fisLabel.read();

				// now, set these up as the input neurons and train...
				int guess = m_nn.testNetwork(imageData);

				// show status
				if(guess == label)
					correct++;							
			}
			System.out.println("TEST RESULTS: " + correct + "/10,000 (" + (100.0 * (double)correct / 10000.0) + "%)");
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
		sn.init();

		// let's train our new network N times
		for(int i = 0; i < TRAINING_EPOCHS; i++) {
			System.out.println("Training ShallowNet, epoch " + i);
			sn.trainNetwork();
		}
		
		// alright, ready to test!
		System.out.println("Testing ShallowNet");
		sn.testNetwork();
		
		// wrap up the show, we're done
		System.out.println("Complete - quitting now");
	}
}
