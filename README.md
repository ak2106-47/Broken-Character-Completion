## Broken Character Completion using Generative Adversarial Networks

Character Recognition is widely used in many day-to-day applications such as conversion of scanned or photographed image of a printed or handwritten document and license plate recognition etc into machine-encoded characters. Often the images acquired are incognizable to the character recognition system due to conditions under which the image was acquired. Motion blur may be caused due to fast movement of the subject of interest, optical blur may occur due to insufficient focusing of the camera. Sometimes the subject of interest may not be clearly visible due to being hidden behind a opaque film such as a coat of mud on license plate during rainy season, or may be worn out due to the wear and tear over the years, such as old documents and historic texts.

One of the approaches to solve this problem is to mend the broken characters before passing them to the character recognition system for further processing. This is achieved using the versatile Generative Adversarial Networks(GAN) which are used extensively in the domain of image processing.

This approach requires for the noisy input image to be segmented into individual characters which may or may not be broken. Then, the GAN repairs it by replacing the corrupted portions of the broken character image so as to recover the clean image of the original character. Then these recovered character images can again be put together to form a clean input image of the subject of interest and then fed to the character recognition system.

