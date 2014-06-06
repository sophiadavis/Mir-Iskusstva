## Standardizes input vectors for use in neural network
images <- read.csv('imageColorData_final.csv')

str(images)
dim(images)

# Standardize a vector of input features to lie in the range [-2, 2]
# Equation comes from this paper: http://www.faqs.org/faqs/ai-faq/neural-nets/part2/
# -- search "Should I standardize the input variables (column vectors)?"
standardize <- function(x) {
	N <- length(x)
	std.x <- numeric(N)
	
	midrange <- (max(x) + min(x))/2
	range <- max(x) - min(x)
	
	std.x <- (x - midrange)/(range/4)
}

stdNumColors <- standardize(images$NumColors)
hist(images$NumColors, breaks = 50, prob = T)
hist(stdNumColors, breaks = 100, prob = T)

imageMatrix <- as.matrix(images[,3:46])
newImageMatrix <- apply(imageMatrix, 2, standardize)

dim(newImageMatrix)
newImageMatrix[,1] == stdNumColors

stdImages <- as.data.frame(cbind(images[,(1:2)], newImageMatrix))

str(stdImages)
dim(stdImages)
stdImages$NumColors == stdNumColors

write.csv(stdImages, row.names = F, file = "imageColorData_final_Standardized.csv")



