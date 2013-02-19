#!/usr/bin/python
import estimation as est
import sampling as sam
import matplotlib.pyplot as plt

if __name__ == "__main__":
    if (raw_input("Compute question 1.1: Plot different gaussians? (Y/n): ") == "Y"):
        plt.figure()
        est.gaussian_plots()
        plt.close()
        print "... Saved result as image gaussian.png"

    print 

    if (raw_input("Compute question 1.2: Random data set? (no will skip 2-7) (Y/n): ") == "Y"):
        data = est.data(100)
        print data
	print
    
        if(raw_input("Compute question 1.3: Maximum likelihood? (Y/n)") == "Y"):
            plt.figure()
            (sample_mean, sample_cov) = est.estimate_params(data)
            print "Sample mean:"
            print sample_mean
            print "Sample cov:"
            print sample_cov
            print "Saved result as image scatter.png"
            plt.close()
        print
    
        if(raw_input("Compute question 1.5: histogram based probability estimate? (Y/n): ") == "Y"):
            plt.figure()
            est.histogram_plot(data, 10)
            print "Saved results as histogram1.png and histogram2.png"
            plt.close()
        print
    
        if(raw_input("Compute question 1.6: Histogram together with analytical solution? (Y/n): ") == "Y"):
            plt.figure()
            est.histogram_and_analytical_plot(data,10)
            plt.close()
            print "Saved results as hist_and_analytical.png"
        print
    
        if(raw_input("Compute question 1.7: 2-dimensional histogram? (Y/n): ") == "Y"):
            plt.figure()
            est.histogram_plot_2d(data)
            plt.close()
            print "Saved results as histogram3d_10bins.png, histogram3d_15bins.png, histogram3d_20bins.png"
    print

    if(raw_input("Compute question 1.8: Monte Carlo? (Y/n): ") == "Y"):
        plt.figure()
        sam.convergence_plot(0.5)
        print "Saved results as montecarlo.png"
        plt.close()
        print

    import objdetection as obj
    import os.path
    if(raw_input("Compute question 1.9: Visualise kandle1 from training set") == "Y"):
        plt.figure()

        if not os.path.exists("data"):
            os.makedirs("data")

        if not (os.path.isfile("data/prob_Z") and os.path.isfile("data/prob_Z_trans")):
            obj.probability_model("kande1.pnm") #This function is very slow
	                                    #delete data folder to rerun it
        obj.display_model(obj.get_Z("data/prob_Z_trans"),(480,640))
        plt.close()

    if(raw_input("Compute question 1.10: Plot object postition and spread? (Y/n): ") == "Y"):
        plt.figure()

	obj.contour_plot(640,480)

	plt.clean()
