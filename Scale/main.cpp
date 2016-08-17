#pragma comment(lib, "opencv_core310d.lib")
#pragma comment(lib, "opencv_highgui310d.lib")
#pragma comment(lib, "opencv_imgproc310d.lib")
#pragma comment(lib, "opencv_imgcodecs310d.lib")

#include <opencv2/opencv.hpp>
using namespace cv;

Mat img;
Mat out_img;

Size DIR_FILTER_SIZE = Size(3, 5);

Size2d covert_params(double upscale_ratio, Size2d in_size)
{
	return Size2d(in_size.width * upscale_ratio, in_size.height*upscale_ratio);
}

void get_kernel_data(Point2d offset, Size dir_filter_size, Mat& kernel_data)
{
	for (int i = 0; i < dir_filter_size.height; i++)
	{
		for (int j = 0; j < dir_filter_size.width; j++)
		{
			continue;
		}
	}
}

void compute_grads(Mat in, Mat& out)
{

	Mat gf, gf2, gf2_ave;
	Mat gr, gr2, gr2_ave;
	Mat g_w, g_w_n;
	double max_val;

	Mat in_y;
	cvtColor(in, in_y, CV_BGR2GRAY);
	in_y.convertTo(in_y, CV_32F);

	float fk[] = { -1, 0, 0, 1 };
	float rk[] = { 0, -1, 1, 0 };

	filter2D(in_y, gf, -1, Mat(2, 2, CV_32F, &fk), Point(-1, -1));
	filter2D(in_y, gr, -1, Mat(2, 2, CV_32F, &rk), Point(-1, -1));

	gf2 = gf.mul(gf);
	gr2 = gr.mul(gr);
	filter2D(gf2, gf2_ave, -1, Mat::ones(DIR_FILTER_SIZE.height, DIR_FILTER_SIZE.width, CV_32F) / DIR_FILTER_SIZE.area(), Point(-1, -1));
	filter2D(gr2, gr2_ave, -1, Mat::ones(DIR_FILTER_SIZE.height, DIR_FILTER_SIZE.width, CV_32F) / DIR_FILTER_SIZE.area(), Point(-1, -1));

	g_w = (gf2_ave - gr2_ave) / (gf2_ave + gr2_ave);
	minMaxLoc(abs(g_w), NULL, &max_val, NULL, NULL);
	g_w_n = g_w / max_val;

	out = g_w_n;
}

void scale(Mat in, Mat& out, Point2d offset, Size2d in_size, Size2d out_size)
{
	Mat out_img(out_size, in.type());
	unsigned char *output = (unsigned char*)(out_img.data); // faster

	double phase_step_x = in_size.width / out_size.width;
	double phase_step_y = in_size.height / out_size.height;

	Point tl = Point2d(floor(offset.x), floor(offset.y));
	Point br = tl + Point(1, 1);
	
	int out_h = 0;
	for (double h = offset.y; h < offset.y + in_size.height; h += phase_step_y)
	{
		int out_w = 0;
		for (double w = offset.x; w < offset.x + in_size.width; w += phase_step_x)
		{

			Point tl = Point2d(floor(w), floor(h));
			Point br = tl + Point(1, 1);

			Point2d local_phase = (offset - Point2d(tl));

			Mat outcolor(1, 1, in.type());
			getRectSubPix(Mat(in, Rect(tl, br)), Size(1,1), local_phase, outcolor);

			output[out_h*(int)(out_size.width*out_img.channels()) + (out_img.channels()*out_w+0)] = outcolor.data[0];
			output[out_h*(int)(out_size.width*out_img.channels()) + (out_img.channels()*out_w+1)] = outcolor.data[1];
			output[out_h*(int)(out_size.width*out_img.channels()) + (out_img.channels()*out_w+2)] = outcolor.data[2];

			out_w++;
		}
		out_h++;
	}

	Mat in_grads;
	//compute_grads(in, in_grads);

	out = out_img;
}

void update_and_show_image(int, void*)
{
	int s = 1;
	scale(img, out_img, Point2d(0, 0), img.size(), img.size() * s);
	imshow("main_window", out_img);
}

void init_params()
{
	DIR_FILTER_SIZE.height = 3;
	DIR_FILTER_SIZE.width = 5;

	setTrackbarPos("filterN_height", "main_window", DIR_FILTER_SIZE.height);
	setTrackbarPos("filterN_width", "main_window", DIR_FILTER_SIZE.width);
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
	if (event == EVENT_LBUTTONDOWN)
	{
		std::cout << "Left button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
	}
	else if (event == EVENT_RBUTTONDOWN)
	{
		std::cout << "Right button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
	}
	else if (event == EVENT_MBUTTONDOWN)
	{
		std::cout << "Middle button of the mouse is clicked - position (" << x << ", " << y << ")" << std::endl;
	}
	else if (event == EVENT_MOUSEMOVE)
	{
		std::cout << "Mouse move over the window - position (" << x << ", " << y << ")" << std::endl;
	}
}

int main(int argc, char** argv)
{
	std::string filename;

	if (argc == 2) {
		filename = argv[1];
	}
	else {
		filename = "D:/Code/Octave/Images/Lenna.png";
	}

	img = imread(filename.c_str(), CV_LOAD_IMAGE_COLOR); // 8-bit n-channel

	if (!img.data) { printf("Error loading image from %s.\n", filename); return -1; }

	namedWindow("main_window", WINDOW_NORMAL | WINDOW_KEEPRATIO);
	resizeWindow("main_window", 600, 600);

	setMouseCallback("main_window", CallBackFunc, NULL);

	// Initialize
	update_and_show_image(0, NULL);

	createTrackbar("filterN_height", "main_window", &DIR_FILTER_SIZE.height, 15, update_and_show_image);
	createTrackbar("filterN_width", "main_window", &DIR_FILTER_SIZE.width, 15, update_and_show_image);

	bool done = false;
	while (!done)
	{
		int c = waitKey(0);
		switch (c)
		{
		case 'q':
			done = true;
			break;
		case 'r':
			init_params();
			update_and_show_image(0, &img);
			break;
		case 'h':
			printf("q - quit\nr = reset/reload\nh - this message\n");
			break;
		}
	}

	destroyWindow("main_window");
	return 0;
}