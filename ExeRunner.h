#pragma once

///most of this file is copied from gams tutorial
///see http://interfaces.gams-software.com/doku.php?id=env:spawning_gams_from_visual_c

#include <exception>
#include <string>
#include <windows.h> 
#include <direct.h>
#include "Exception.h"
using namespace std;

//runtime exception
class ExeRunnerException : Exception
{ 
public:
	ExeRunnerException(string text) : Exception(text){
	}
};

class ExeRunner
{
public:
	static string ExeRunner::RunExe(const string &exePath, const string &params) {
		string outputData; //app output

		/////////////////////////////////////////////////////////////////////////////
		// See Microsoft publication:
		// HOWTO: Spawn Console Processes with Redirected Standard Handles (Q190351)
		/////////////////////////////////////////////////////////////////////////////

		HANDLE hChildProcess = NULL;

		//
		// set up security attributes
		//
		SECURITY_ATTRIBUTES sa;
		sa.nLength = sizeof(SECURITY_ATTRIBUTES);
		sa.lpSecurityDescriptor = NULL;
		sa.bInheritHandle = TRUE;

		//
		// create child output pipe
		//
		HANDLE hOutputReadTmp;
		HANDLE hOutputWrite;
		if (CreatePipe(&hOutputReadTmp, &hOutputWrite, &sa, 0) == 0) {
			throw ExeRunnerException("CreatePipe");
		}

		//
		// Create a duplicate of the output write handle for the std error
		// write handle. This is necessary in case the child application
		// closes one of its std output handles.
		//
		HANDLE hErrorWrite;
		if (DuplicateHandle(GetCurrentProcess(),hOutputWrite,
			GetCurrentProcess(),&hErrorWrite,0,
			TRUE,DUPLICATE_SAME_ACCESS) == 0) {
				throw ExeRunnerException("DuplicateHandle");
		}

		//
		// Create the child input pipe.
		//
		HANDLE hInputRead;
		HANDLE hInputWriteTmp;
		if (CreatePipe(&hInputRead,&hInputWriteTmp,&sa,0) == 0) {
			throw ExeRunnerException ("CreatePipe");
		}


		//
		// Create new output read handle and the input write handles. Set 
		// the Properties to FALSE. Otherwise, the child inherits the
		// properties and, as a result, non-closeable handles to the pipes
		// are created.
		//
		HANDLE hOutputRead;
		HANDLE hInputWrite;

		if (DuplicateHandle(GetCurrentProcess(),hOutputReadTmp,
			GetCurrentProcess(),
			&hOutputRead, // Address of new handle.
			0,FALSE, // Make it uninheritable.
			DUPLICATE_SAME_ACCESS) == 0) {
				throw ExeRunnerException("DupliateHandle");
		}

		if (DuplicateHandle(GetCurrentProcess(),hInputWriteTmp,
			GetCurrentProcess(),
			&hInputWrite, // Address of new handle.
			0,FALSE, // Make it uninheritable.
			DUPLICATE_SAME_ACCESS) == 0) {
				throw ExeRunnerException ("DupliateHandle");
		}

		//
		// Close inheritable copies of the handles you do not want to be
		// inherited.
		//
		if (!CloseHandle(hOutputReadTmp)) {
			throw ExeRunnerException("CloseHandle");
		}


		if (!CloseHandle(hInputWriteTmp)) {
			throw ExeRunnerException("CloseHandle");
		}

		PROCESS_INFORMATION pi;
		STARTUPINFO si;

		//
		// call CreateProcess
		//
		ZeroMemory(&si,sizeof(STARTUPINFO));
		si.cb = sizeof(STARTUPINFO);
		si.dwFlags = STARTF_USESTDHANDLES | STARTF_USESHOWWINDOW;
		si.hStdOutput = hOutputWrite;
		si.hStdInput  = hInputRead;
		si.hStdError  = hErrorWrite;
		si.wShowWindow = SW_HIDE;
		// Use this if you want to hide the child:
		//     si.wShowWindow = SW_HIDE;
		// Note that dwFlags must include STARTF_USESHOWWINDOW if you want to
		// use the wShowWindow flags.

		//
		// compose command line
		// if components contain a blank protect by double quotes
		//
		string cmdLine;
		cmdLine = "\"" + exePath + "\"" + " " + params;
		LPSTR cmdLineLPSTR = _strdup(cmdLine.c_str());

		// Launch the process that you want to redirect (in this case,
		// Child.exe). Make sure Child.exe is in the same directory as
		// redirect.c launch redirect from a command line to prevent location
		// confusion.
		if (!CreateProcess(
			exePath.c_str(),  // pointer to name of executable module
			cmdLineLPSTR,  // pointer to command line string
			NULL,     // process security attributes
			NULL,     // thread security attributes
			TRUE,     // handle inheritance flag
			CREATE_NEW_CONSOLE,  // creation flags
			NULL,  // pointer to new environment block
			NULL,  // pointer to current directory name
			&si,   // pointer to STARTUPINFO
			&pi    // pointer to PROCESS_INFORMATION
			)) {
				free(cmdLineLPSTR);
				throw ExeRunnerException("CreateProcess");
		}

		free(cmdLineLPSTR);

		// Set global child process handle to cause threads to exit.
		hChildProcess = pi.hProcess;

		// Close any unnecessary handles.
		if (!CloseHandle(pi.hThread)) 
			throw ExeRunnerException("CloseHandle");

		// Close pipe handles (do not continue to modify the parent).
		// You need to make sure that no handles to the write end of the
		// output pipe are maintained in this process or else the pipe will
		// not close when the child process exits and the ReadFile will hang.
		if (!CloseHandle(hOutputWrite)) throw ExeRunnerException("CloseHandle");
		if (!CloseHandle(hInputRead )) throw ExeRunnerException("CloseHandle");
		if (!CloseHandle(hErrorWrite)) throw ExeRunnerException("CloseHandle");

		// Read and handle app output
		CHAR lpBuffer[256];
		DWORD nBytesRead;

		while(TRUE)
		{
			if (!ReadFile(hOutputRead,lpBuffer,sizeof(lpBuffer),
				&nBytesRead,NULL) || !nBytesRead)
			{
				if (GetLastError() == ERROR_BROKEN_PIPE)
					break; // pipe done - normal exit path.
				else
					throw ExeRunnerException("ReadFile"); // Something bad happened.
			}

			//
			// here we have data from stdout. copy it to the string
			//
			string s(lpBuffer, nBytesRead);
			outputData += s;
		}

		if (WaitForSingleObject(hChildProcess,INFINITE) == WAIT_FAILED)
			throw ExeRunnerException("WaitForSingleObject");

		if (!CloseHandle(hOutputRead)) throw ExeRunnerException("CloseHandle");
		if (!CloseHandle(hInputWrite)) throw ExeRunnerException("CloseHandle");

		return outputData;
	}
};
