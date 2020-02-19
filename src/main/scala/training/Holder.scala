package training

import org.apache.log4j.Logger

object Holder {
  @transient lazy val log: Logger = Logger.getLogger(getClass.getName)
}
