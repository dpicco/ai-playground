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

	// let's initialize our network
	public void init() {
		
		// let's create and initialize our NeuralNetwork container
		m_nn = new NeuralNetwork();
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
/*				if(i % 10000 == 0) {
					System.out.println("Neural Network Status, Training run " + i);
					System.out.println("Ideal: " + label + ", Actual: " + guess);					
					m_nn.printNetwork();
				}
*/
			}
			System.out.println("Result: " + correct + "/60,000 (" + (100 * correct / 60000) + "%)");
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

		// let's train our new network 10 times
		for(int i = 0; i < 1000; i++) {
			System.out.println("Training ShallowNet, epoch " + i);
			sn.trainNetwork();
		}
		
		// wrap up the show, we're done
		System.out.println("Complete - quitting now");
	}
}
