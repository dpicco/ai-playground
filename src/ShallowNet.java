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

	// This is our multi-layer capable neural network object
	private MLNeuralNetwork m_mlnn = null;
	
	// number of training epochs
	private static int TRAINING_EPOCHS = 20;

	// let's initialize our network
	public void init() {
		
		// let's create and initialize our NeuralNetwork container
		m_mlnn = new MLNeuralNetwork(new int[]{28*28, 32, 16, 10}, 0.2);
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
				m_mlnn.trainNetwork(imageData, label);

				// show status
				int guess = m_mlnn.labelFromVector(m_mlnn.vectorFromOutput());
				if(guess == label)
					correct++;				
				
				// dump a status of the network once in a while
				if(i % 6000 == 0) {
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
				int guess = m_mlnn.testNetwork(imageData);

				// show status
				if(guess == label)
					correct++;							

				// dump a status of the network once in a while
				if(i % 1000 == 0) {
					System.out.println((100.0 * (double)i / 10000.0) + "%... ");
				}
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

	// output our numbers to a file
	private void imagineNumbers() {
	
		m_mlnn.reverseActivateImage("imagine_0.png", 0);
		m_mlnn.reverseActivateImage("imagine_1.png", 1);
		m_mlnn.reverseActivateImage("imagine_2.png", 2);
		m_mlnn.reverseActivateImage("imagine_3.png", 3);
		m_mlnn.reverseActivateImage("imagine_4.png", 4);
		m_mlnn.reverseActivateImage("imagine_5.png", 5);
		m_mlnn.reverseActivateImage("imagine_6.png", 6);
		m_mlnn.reverseActivateImage("imagine_7.png", 7);
		m_mlnn.reverseActivateImage("imagine_8.png", 8);
		m_mlnn.reverseActivateImage("imagine_9.png", 9);
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
			System.out.println("Training MLNeuralNetwork, epoch " + (i+1) + " of " + TRAINING_EPOCHS);
			sn.trainNetwork();
		}
		
		// alright, ready to test!
		System.out.println("Testing MLNeuralNetwork");
		sn.testNetwork();
		
		// now let's see what the machine imagines
		sn.imagineNumbers();
		
		// wrap up the show, we're done
		System.out.println("Complete - quitting now");
	}
}
