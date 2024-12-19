package ddcm.ugent.be.exceptions;

public class NoMatchException extends RuntimeException {

    public NoMatchException(String message) {
        super(message);
    }

    public NoMatchException(String message, Throwable cause) {
        super(message, cause);
    }

    public NoMatchException(Throwable cause) {
        super(cause);
    }

    public NoMatchException(String message, Throwable cause, boolean enableSuppression, boolean writableStackTrace) {
        super(message, cause, enableSuppression, writableStackTrace);
    }

}
