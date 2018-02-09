
//****************************************************************************
// Also note that we've supplied a helpful debugging function called checkCudaErrors.
// You should wrap your allocation and copying statements like we've done in the
// code we're supplying you. Here is an example of the unsafe way to allocate
// memory on the GPU:
//
// cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols);
//
// Here is an example of the safe way to do the same thing:
//
// checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRows * numCols));
//****************************************************************************

#include <iostream>
#include <iomanip>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

template<typename T>
void check(T err, const char* const func, const char* const file, const int line) {
	if (err != cudaSuccess) {
		std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
		std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
		exit(1);
	}
}

__global__
void box_filter(const unsigned char* const inputChannel,
	unsigned char* const outputChannel,
	int numRows, int numCols,
	const float* const filter, const int filterWidth)
{
	// TODO: 
	// NOTA: Cuidado al acceder a memoria que esta fuera de los limites de la imagen
	//
	// if ( absolute_image_position_x >= numCols ||
	//      absolute_image_position_y >= numRows )
	// {
	//     return;
	// }
	// NOTA: Que un thread tenga una posición correcta en 2D no quiere decir que al aplicar el filtro
	// los valores de sus vecinos sean correctos, ya que pueden salirse de la imagen.

	//Acceso a los pixeles mediante los threads del mismo modo que el kernel dado en la practica
	 const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
	blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	//Creo las variables oportunas para la realizacion del ejercicio
	int i = 0;
	int j = 0;
	int mitad = (filterWidth - 1) / 2; //Variable necesaria para poder movernos dentro de la "matriz" del filtro
									   //Y para ayudarnos a saber si la posicion del pixel que se recorre dentro de los FOR
									   //Esta dentro de la propia imagen
	float pixel = 0; //Color del canal

	//Realizamos dos bucles for anidados para poder acceder a toda la imagen
	for (i = 0; i < filterWidth; i++)
	{
		for (j = 0; j < filterWidth; j++)
		{
			int v_posY = thread_2D_pos.y + j - mitad; //Calculamos la posicion del pixel en Y
			int v_posX = thread_2D_pos.x + i - mitad; //Calculamos la posocion del pixel en X
			/*if (v_posX < 0) //Si la posicion actual en X esta fuera (por la izquierda) de la imagen, se "fuerza" a que el pixel sea el borde
				v_posX = 0;
			if (v_posY < 0) //Si la posicion actual en Y esta fuera (por arriba) de la imagen, se "fuerza" a que el pixel sea el borde
				v_posY = 0;
			if (v_posX >= numCols) //Si la posicion actual en X esta fuera (por la derecha) de la imagen, se "fuerza" a que el pixel sea el borde
				v_posX = numCols - 1;
			if (v_posY >= numRows) //Si la posicion actual en X esta fuera (por la abajo) de la imagen, se "fuerza" a que el pixel sea el borde
				v_posY = numRows - 1;
			*/
			
			v_posY = min(max(v_posY, 0), numRows - 1); //La posicion Y se fuerza a que este dentro de la imagen
			v_posX = min(max(v_posX, 0), numCols - 1); //La posicion X se fuerza a que este dentro de la imagen

			float v_filtro = filter[(i)+(j)*filterWidth]; //Cogemos el valor del filtro correspondiente
			int v_pos = inputChannel[v_posY * numCols + v_posX]; //Cogemos el valor del pixel correspondiente
			pixel += v_filtro * v_pos; //Vamos acumulando todas las multiplicaciones en la variable pixel
		}
	}
	outputChannel[thread_1D_pos] = pixel; //Y finalmente añadimos en la posicion central del thread (el de entrada) en la imagen
										  //el valor de todas las multiplicaciones
	
	//----CODIGO DE MEMORIA COMPARTIDA----
	//No esta completo, ya que no me terminaba de funcionar, pero he considerado oportuno dejarlo comentado
	//Para que veais lo que he podido avanzar en este ejercicio

	/*
	int i = 0;
	int j = 0;
	int mitad = (filterWidth - 1) / 2;
	float pixel = 0;
	__shared__ float m_g[32*32];
	__shared__ float f_g[5*5];
	
	m_g[threadIdx.y * 32 + threadIdx.x] = inputChannel[thread_1D_pos];
	__syncthreads();

	for (int i = 0; i < 5*5; i++) {
		f_g[i] = filter[i];
	}
	__syncthreads();

	for (i = 0; i < filterWidth; i++)
	{
		for (j = 0; j < filterWidth; j++)
		{
			pixel += f_g[(i)+(j)*filterWidth] * m_g[(threadIdx.y+i-mitad) + (threadIdx.x + j - mitad)];

		}
	}
	outputChannel[thread_1D_pos] = pixel;
	*/

}

//This kernel takes in an image represented as a uchar4 and splits
//it into three images consisting of only one color channel each
__global__
void separateChannels(const uchar4* const inputImageRGBA,
	int numRows,
	int numCols,
	unsigned char* const redChannel,
	unsigned char* const greenChannel,
	unsigned char* const blueChannel)
{
	// TODO: 
	// NOTA: Cuidado al acceder a memoria que esta fuera de los limites de la imagen

	// if ( absolute_image_position_x >= numCols ||
	//      absolute_image_position_y >= numRows )
	// {
	//     return;
	// }
	//const int myRow = blockIdx.x*blockDim.x + threadIdx.x;
	//const int myCol = blockIdx.y*blockDim.y + threadIdx.y;
	//printf("HOLA-- %s\n" ,(const char *)myRow);

	//Acceso a los pixeles mediante los threads del mismo modo que el kernel dado en la practica
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);
	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	//Accediendo al pixel (uchar4) mediante el thread correspondiente, devolvemos la posicion XYZ (RGB) a su canal correcto.
	redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
	greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
	blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;
}

//This kernel takes in three color channels and recombines them
//into one image. The alpha channel is set to 255 to represent
//that this image has no transparency.
__global__
void recombineChannels(const unsigned char* const redChannel,
	const unsigned char* const greenChannel,
	const unsigned char* const blueChannel,
	uchar4* const outputImageRGBA,
	int numRows,
	int numCols)
{
	const int2 thread_2D_pos = make_int2(blockIdx.x * blockDim.x + threadIdx.x,
		blockIdx.y * blockDim.y + threadIdx.y);

	const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

	//make sure we don't try and access memory outside the image
	//by having any threads mapped there return early
	if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
		return;

	unsigned char red = redChannel[thread_1D_pos];
	unsigned char green = greenChannel[thread_1D_pos];
	unsigned char blue = blueChannel[thread_1D_pos];

	//Alpha should be 255 for no transparency
	uchar4 outputPixel = make_uchar4(red, green, blue, 255);

	outputImageRGBA[thread_1D_pos] = outputPixel;
}

unsigned char *d_red, *d_green, *d_blue;
float         *d_filter;

void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage,
	const float* const h_filter, const size_t filterWidth)
{

	//allocate memory for the three different channels
	checkCudaErrors(cudaMalloc(&d_red, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage));
	checkCudaErrors(cudaMalloc(&d_blue, sizeof(unsigned char) * numRowsImage * numColsImage));

	//TODO:
	//Reservar memoria para el filtro en GPU: d_filter, la cual ya esta declarada

	//En un principio unicamente reservaba memoria de tamaño de filtro, por lo que no me funcionaba correctamente.
	//Lo corregí adjudicando una memoria del doble, al ser una matriz
	checkCudaErrors(cudaMalloc(&d_filter, sizeof(float) * filterWidth*filterWidth));
	// Copiar el filtro  (h_filter) a memoria global de la GPU (d_filter)
	checkCudaErrors(cudaMemcpy(d_filter, h_filter, sizeof(float)* filterWidth*filterWidth, cudaMemcpyHostToDevice));
}


void create_filter(float **h_filter, int *filterWidth) {
	
	/*
	//------------FILTRO GAUSSIANO: BLUR------------//
	const int KernelWidth = 5; //OJO CON EL TAMAÑO DEL FILTRO//
	*filterWidth = KernelWidth;
	*h_filter = new float[KernelWidth * KernelWidth];

	//Filtro gaussiano: blur
	const float KernelSigma = 2.;
	float filterSum = 0.f; //for normalization

	for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) {
	for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
	float filterValue = expf( -(float)(c * c + r * r) / (2.f * KernelSigma * KernelSigma));
	(*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] = filterValue;
	filterSum += filterValue;
	}
	}

	float normalizationFactor = 1.f / filterSum;

	for (int r = -KernelWidth/2; r <= KernelWidth/2; ++r) {
	for (int c = -KernelWidth/2; c <= KernelWidth/2; ++c) {
	(*h_filter)[(r + KernelWidth/2) * KernelWidth + c + KernelWidth/2] *= normalizationFactor;
	}
	}
	*/

	//------------FILTRO LAPLACIANO 5X5------------//
	
	const int KernelWidth = 5; //OJO CON EL TAMAÑO DEL FILTRO//
	*filterWidth = KernelWidth;
	*h_filter = new float[KernelWidth * KernelWidth];
	(*h_filter)[0] = 0;   (*h_filter)[1] = 0;    (*h_filter)[2] = -1.;  (*h_filter)[3] = 0;    (*h_filter)[4] = 0;
	(*h_filter)[5] = 1.;  (*h_filter)[6] = -1.;  (*h_filter)[7] = -2.;  (*h_filter)[8] = -1.;  (*h_filter)[9] = 0;
	(*h_filter)[10] = -1.; (*h_filter)[11] = -2.; (*h_filter)[12] = 17.; (*h_filter)[13] = -2.; (*h_filter)[14] = -1.;
	(*h_filter)[15] = 1.; (*h_filter)[16] = -1.; (*h_filter)[17] = -2.; (*h_filter)[18] = -1.; (*h_filter)[19] = 0;
	(*h_filter)[20] = 0;  (*h_filter)[21] = 0;   (*h_filter)[22] = -1.; (*h_filter)[23] = 0;   (*h_filter)[24] = 0;
	
	//TODO: crear los filtros segun necesidad
	//NOTA: cuidado al establecer el tamaño del filtro a utilizar

	//------------FILTRO NITIDEZ 3X3------------//
	/*
	const int KernelWidth = 3; //OJO CON EL TAMAÑO DEL FILTRO//
	*filterWidth = KernelWidth;
	*h_filter = new float[KernelWidth * KernelWidth];
	(*h_filter)[0] = 0;   (*h_filter)[1] = -0.25;    (*h_filter)[2] = 0;
	(*h_filter)[3] = -0.25;  (*h_filter)[4] = 2.0;  (*h_filter)[5] = -0.25;
	(*h_filter)[6] = 0; (*h_filter)[7] = -0.25; (*h_filter)[8] = 0;
	*/

	//------------FILTRO DETECCION DE BORDES 3X3------------//
	/*
	const int KernelWidth = 3; //OJO CON EL TAMAÑO DEL FILTRO//
	*filterWidth = KernelWidth;
	*h_filter = new float[KernelWidth * KernelWidth];
	(*h_filter)[0] = 0;   (*h_filter)[1] = 1;    (*h_filter)[2] = 0; 
	(*h_filter)[3] = 1;  (*h_filter)[4] = -4.0;  (*h_filter)[5] = 1; 
	(*h_filter)[6] = 0; (*h_filter)[7] = 1; (*h_filter)[8] = 0; 
	*/

	//------------FILTRO SUAVIZADO 3X3------------//
	/*
	const int KernelWidth = 3; //OJO CON EL TAMAÑO DEL FILTRO//
	*filterWidth = KernelWidth;
	*h_filter = new float[KernelWidth * KernelWidth];
	(*h_filter)[0] = 0.111;   (*h_filter)[1] = 0.111;    (*h_filter)[2] = 0.111;
	(*h_filter)[3] = 0.111;  (*h_filter)[4] = 0.111;  (*h_filter)[5] = 0.111;
	(*h_filter)[6] = 0.111; (*h_filter)[7] = 0.111; (*h_filter)[8] = 0.111;
	*/
	
}


void convolution(const uchar4 * const h_inputImageRGBA, uchar4 * const d_inputImageRGBA,
	uchar4* const d_outputImageRGBA, const size_t numRows, const size_t numCols,
	unsigned char *d_redFiltered,
	unsigned char *d_greenFiltered,
	unsigned char *d_blueFiltered,
	const int filterWidth) 
{
	//TODO: Calcular tamaños de bloque
	//const dim3 gridSize((numCols - 1) / filterWidth + 1, (numRows - 1) / filterWidth + 1, 1);
	//const dim3 blockSize(filterWidth, filterWidth, 1);
	//Despues de probar con varios tamaños de bloque, he decidido darle 32 debido a su mejor rendimiento.
	//Anteriormente habia probado con un tamaño igual que el filtro, pero era bastante pobre de rendimiento.

	int tam_block = 32;
	const dim3 blockSize(tam_block, tam_block, 1);
	//GridSize calculado en funcion del tamaño de la imagen introducida, como se pide en la memoria
	const dim3 gridSize((numCols-1) / tam_block + 1, (numRows-1) / tam_block + 1, 1);


	//TODO: Lanzar kernel para separar imagenes RGBA en diferentes colores
	//Lanzamos el kernel separateChannels con el grid y block correspondiente
	separateChannels <<<gridSize, blockSize >> > (d_inputImageRGBA, numRows, numCols, d_red, d_green, d_blue);

	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); //Funcion de sincronización y devuelve errores

	//TODO: Ejecutar convolución. Una por canal
	//Lanzamos el kernel box_filter una vez por cada canal de color,  con el grid y block correspondiente
	box_filter <<<gridSize, blockSize >> > (d_red, d_redFiltered, numRows, numCols, d_filter, filterWidth);
	box_filter <<<gridSize, blockSize >> > (d_green, d_greenFiltered, numRows, numCols, d_filter, filterWidth);
	box_filter <<<gridSize, blockSize >> > (d_blue, d_blueFiltered, numRows, numCols, d_filter, filterWidth);

	//Funcion de sincronización y devuelve errores
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

	// Recombining the results. 
	recombineChannels << <gridSize, blockSize >> >(d_redFiltered, d_greenFiltered, d_blueFiltered, d_outputImageRGBA, numRows, numCols);

	//Funcion de sincronización y devuelve errores
	cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError()); 

}


//Free all the memory that we allocated
//TODO: make sure you free any arrays that you allocated
void cleanup() {
	checkCudaErrors(cudaFree(d_red));
	checkCudaErrors(cudaFree(d_green));
	checkCudaErrors(cudaFree(d_blue));
	checkCudaErrors(cudaFree(d_filter)); //Liberamos memoria del filtro
}
