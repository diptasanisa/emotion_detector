self.line.set_xdata(range(len(categories)))
        self.line.set_ydata(percentages)
        self.ax.set_xticks(range(len(categories)))
        self.ax.set_xticklabels(categories, rotation=45, ha='right')
        self.ax.set_yticks(np.arange(0, 101, 10))  # Sumbu y dari 0 hingga 100 dengan interval 10
        self.ax.set_ylim(0, 100)  # Batas sumbu y dari 0 hingga 100
        self.canvas_chart.draw()