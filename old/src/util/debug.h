/* Copyright (C) 2013 Lasse Schuirmann

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 * @file debug_message_settings.h
 * @author Lasse Schuirmann <lasse.schuirmann@gmail.com>
 * 
 * @brief This file provides the possibility to activate various trace outputs.
 * Feel free to add more.
 * 
 * Here is an example how a debug output within the program may look like:
 * 
 *  #if VERB_TYPE_ACTIVE(PARSE_TRACE)\n
 *    DEBUG_OUTPUT(PARSE_TRACE_STR << "Descriptive text " << some_var);\n
 *  #endif \n
 * 
 * The \\n will be added automatically.
 */

#if !defined (debug_h)
#define debug_h

#include <iostream>
#include <time.h>

/**
 * Defines what types of debug output may occur
 */
#define VERBOSE           (ALL)
/// If trace is not 0, filename and line will be added to the output
#define TRACE             1
/// Output stream
#define OUTPUT_STREAM     cerr
/// Prefix which is put out before every debug output
#define OUTP_PREFIX       "[DBG]"

/**
 * Deactivates all debug output
 * 
 * If VERBOSE is NONE, VERB_TYPE_ACTIVE(arg) will return false for every arg.
 */
#define NONE              0
#define NONE_STR          "[NON] "
/**
 * Prints out some information about the platform.
 */
#define PLATFORM_INFO     1
#define PLATFORM_INFO_STR "[PLA] "
/**
 * Prints out some information about the device.
 */
#define DEVICE_INFO       2
#define DEVICE_INFO_STR   "[DEV] "

/* Define new ones here.
 * Just take the next bit - if you reach 0x80, be sure to redefine ALL and
 * everything should work as before.
 */
/// Activates all output types
#define ALL               0xFF

/**
 * VERB_TYPE_ACTIVE
 * 
 * Returns false/0 if the given verbose type is inactive. If it's active, the
 * type value is returned.
 * 
 * @param type The verbose type to check.
 * @return boolean
 */
#define VERB_TYPE_ACTIVE(type) (((VERBOSE) & (type)) != 0)


/**
 * DEBUG_OUTPUT
 * 
 * Outputs the given string as debug output. It adds some more information
 * which may be configured in debug_message_settings.h.
 * 
 * Note that the brackets around output are omitted intentionally to allow
 * passing multiple variables.
 * 
 * @param output the string to output
 */
#if ((TRACE) != 0)
#define DEBUG_OUTPUT(output)  do { \
  std::OUTPUT_STREAM << OUTP_PREFIX << output << std::endl;\
  std::OUTPUT_STREAM << "     File: " << __FILE__ << "\n     Line: " << __LINE__ << \
  std::endl;\
} while(0)
#else
#define DEBUG_OUTPUT(output)  std::OUTPUT_STREAM << OUTP_PREFIX << output << std::endl;
#endif

#endif /* debug_h */

