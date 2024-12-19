package ddcm.ugent.be.metrics;

public class RepairMetrics  {

    private final long errors;
    private final long positives;
    private final long truePositives;

    public RepairMetrics(long errors, long positives, long truePositives) {
        this.errors = errors;
        this.positives = positives;
        this.truePositives = truePositives;
    }

    /**
     * Get number of errors
     * @return number of errors
     */
    public long getErrors() {
        return errors;
    }

    /**
     * Get number of positives
     * @return number of positives
     */
    public long getPositives() {
        return positives;
    }

    /**
     * Get number of true positives
     * @return number of true positives
     */
    public long getTruePositives() {
        return truePositives;
    }

   
    public double getPrecision() {
        return (double) truePositives / (double) positives;
    }

    public double getRecall() {
        return (double) truePositives / (double) errors;
    }
    
    public double getF1(){
        return 2 * ((double) getPrecision() * (double) getRecall())/((double) getPrecision() + (double) getRecall());
    }


    public String toString() {
        return "RepairMetrics{" +
                "errors=" + errors +
                ", positives=" + positives +
                ", truePositives=" + truePositives +
                '}';
    }
}
