Tried to make the code AD-friendly, however it is not yet possible as arrays are immutable objects in Zygote. Second objective was to have an archive structure were data to be plotted will be saved, instead of within the agent structures. This would make agent structures easier to implement in multi-state environments and the code cleaner. 

Running agents in environments is operational in this branch. Plotting and optimisation not currently working.
