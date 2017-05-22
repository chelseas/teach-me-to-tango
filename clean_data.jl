include("clean_helper_functions.jl")
using PyPlot


# Load data - this is slow.
t, tra_tango, tra_vicon = load_data()

# Estimate error rates
error, error_rate = estimate_error(t, tra_tango, tra_vicon)

# Show XYZ data along
if(false)
figure(1); clf()
ax = ["x","y","z"]
for k=1:2
    subplot(2,1,k)
    title("Translation $(ax[k])")
    if(k==2)
      plot(t, tra_tango[:,3])
    elseif(k==3)
      plot(t,tra_tango[:,2])
    else
      plot(t,tra_tango[:,1])
    end
    plot(t, tra_vicon[:,k])
    legend(["Tango $(ax[k])", "Vicon $(ax[k])"],loc="lower left")
    ylabel("position")
    xlabel("Time (s)")
end
end
# Show x data, error rates
figure(3); clf()

kmax=1
for k=1:kmax
        subplot(2,kmax,2*(k-1)+1);
        if(k==2)
          plot(t, tra_tango[:,3])
        elseif(k==3)
          plot(t, tra_tango[:,2])
        else
          plot(t,tra_tango[:,1])
        end
        plot(t, tra_vicon[:,k])
        ylabel("$(ax[k]) position (m)")
        xlabel("Time (s)")
        legend(["Tango", "Vicon"],loc="lower left")

        subplot(2,kmax,2*k);
        plot(t, error_rate[:,k])
        ylabel("$(ax[k]) position error rate (m/s)")
        xlabel("Time (s)")
end
