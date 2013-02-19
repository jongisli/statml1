#!/usr/bin/python
import estimation as est
import sampling as sam
import matplotlib.pyplot as plt

def main():    
    img_format = "png" 

    if (raw_input("Compute question 1.1: Plot different gaussians? (y/n): ") == "y"):
        plt.figure()
        est.gaussian_plots()
        plt.close()
        print "... Saved result as image gaussian.%s" % img_format
    print 

    if (raw_input("Compute question 1.2: Random data set? (no will skip 2-7) (y/n): ") == "y"):
        data = est.data(100)
        print data
	print
    
        if(raw_input("Compute question 1.3: Maximum likelihood? (y/n)") == "y"):
            plt.figure()
            (sample_mean, sample_cov) = est.estimate_params(data)
            print "Sample mean:"
            print sample_mean
            print "Sample cov:"
            print sample_cov
            print "Saved result as image scatter.%s" % img_format
            plt.close()
        print
    
        if(raw_input("Compute question 1.5: histogram based probability estimate? (y/n): ") == "y"):
            plt.figure()
            est.histogram_plot(data, 10)
            print "Saved results as histogram1.png and histogram2.%s" % img_format
            plt.close()
        print
    
        if(raw_input("Compute question 1.6: Histogram together with analytical solution? (y/n): ") == "y"):
            plt.figure()
            est.histogram_and_analytical_plot(data,10)
            plt.close()
            print "Saved results as hist_and_analytical.%s" % img_format
        print
    
        if(raw_input("Compute question 1.7: 2-dimensional histogram? (y/n): ") == "y"):
            plt.figure()
            est.histogram_plot_2d(data)
            plt.close()
            print "Saved results as histogram3d_10bins.png, histogram3d_15bins.png, histogram3d_20bins.%s" % img_format
    print

    if(raw_input("Compute question 1.8: Monte Carlo? (y/n): ") == "y"):
        plt.figure()
        sam.convergence_plot(0.5)
	sam.convergence_log_plot(0.5)
        print "Saved results as montecarlo.%s" % img_format
        print "Saved results as montecarlo_logplot.%s" % img_format
        plt.close()
    print
    
    import objdetection as obj
    import os.path
    
    if(raw_input("Compute probability model for kande1.pnm? (no will skip 9-11) (y/n): ") == "y"):
        mu,sigma = obj.probability_model("kande1.pnm")
	print
    else:
        return

    if(raw_input("Compute question 1.9: Visualise kande1 probability density? (y/n): ") == "y"):
        plt.figure()

        if not os.path.exists("data"):
            os.makedirs("data")

        if not (os.path.isfile("data/Z_kande1.pnm.data") and os.path.isfile("data/Z_trans_kande1.pnm.data")):
            obj.density_estimate("kande1.pnm",mu,sigma) #This function is very slow
	                                       #delete data folder files rerun it
        obj.display_model("kande1.pnm")
        print "Saved results as kande1_density.png"
        plt.close()
        print

    if(raw_input("Compute question 1.10: Plot object postition and spread of kande1? (y/n): ") == "y"):
        plt.figure()

	obj.contour_plot("kande1.pnm")
        print "Saved results as kande1_and_contours.png"
        plt.close()
        print

    if(raw_input("Compute question 1.11: Visualise kande2 probability density? (y/n): ") == "y"):
        plt.figure()

        if not os.path.exists("data"):
            os.makedirs("data")
            
        if not (os.path.isfile("data/Z_kande2.pnm.data") and os.path.isfile("data/Z_trans_kande2.pnm.data")):
            obj.density_estimate("kande2.pnm",mu,sigma) #This function is very slow
	                                       #delete data folder files rerun it
        obj.display_model("kande2.pnm")
        print "Saved results as kande2_density.png"
        plt.close()
        print

    if(raw_input("Compute question 1.11: Plot object postition and spread of kande2? (y/n): ") == "y"):
        plt.figure()
	obj.contour_plot("kande2.pnm")
        print "Saved results as kande2_and_contours.png"
        plt.close()
        print

if __name__ == "__main__":
    main()
