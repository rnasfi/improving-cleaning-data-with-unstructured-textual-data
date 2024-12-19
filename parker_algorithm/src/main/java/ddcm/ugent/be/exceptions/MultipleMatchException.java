package ddcm.ugent.be.exceptions;

public class MultipleMatchException extends RuntimeException {

    public MultipleMatchException(String message) {
        super(message);
    }

    public MultipleMatchException(String message, Throwable cause) {
        super(message, cause);
    }

    public MultipleMatchException(Throwable cause) {
        super(cause);
    }

    public MultipleMatchException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }

}
