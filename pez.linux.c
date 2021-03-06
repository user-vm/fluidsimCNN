// Pez was developed by Philip Rideout and released under the MIT License.

#include "pez.h"
#include "bstrlib.h"
#include <sys/time.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <signal.h>
#include <wchar.h>
#include <Xm/MwmUtil.h>

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xmd.h>

typedef struct PlatformContextRec
{
    Display* MainDisplay;
    Window MainWindow;
} PlatformContext;

unsigned int GetMicroseconds()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return tp.tv_sec * 1000000 + tp.tv_usec;
}

int main(int argc, char** argv)
{
    int attrib[] = {
        GLX_RENDER_TYPE, GLX_RGBA_BIT,
        GLX_DRAWABLE_TYPE, GLX_WINDOW_BIT,
        GLX_DOUBLEBUFFER, True,
        GLX_RED_SIZE, 8,
        GLX_GREEN_SIZE, 8,
        GLX_BLUE_SIZE, 8,
        GLX_ALPHA_SIZE, 8,
        GLX_DEPTH_SIZE, 24,
        None
    };

    PlatformContext context;

    context.MainDisplay = XOpenDisplay(NULL);
    int screenIndex = DefaultScreen(context.MainDisplay);
    Window root = RootWindow(context.MainDisplay, screenIndex);

    int fbcount;
    PFNGLXCHOOSEFBCONFIGPROC glXChooseFBConfig = (PFNGLXCHOOSEFBCONFIGPROC)glXGetProcAddress((GLubyte*)"glXChooseFBConfig");
    GLXFBConfig *fbc = glXChooseFBConfig(context.MainDisplay, screenIndex, attrib, &fbcount);
    if (!fbc)
        pezFatal("Failed to retrieve a framebuffer config\n");

    PFNGLXGETVISUALFROMFBCONFIGPROC glXGetVisualFromFBConfig = (PFNGLXGETVISUALFROMFBCONFIGPROC) glXGetProcAddress((GLubyte*)"glXGetVisualFromFBConfig");
    if (!glXGetVisualFromFBConfig)
        pezFatal("Failed to get a GLX function pointer\n");

    PFNGLXGETFBCONFIGATTRIBPROC glXGetFBConfigAttrib = (PFNGLXGETFBCONFIGATTRIBPROC) glXGetProcAddress((GLubyte*)"glXGetFBConfigAttrib");
    if (!glXGetFBConfigAttrib)
        pezFatal("Failed to get a GLX function pointer\n");

    if (PezGetConfig().Multisampling) {
        int best_fbc = -1, worst_fbc = -1, best_num_samp = -1, worst_num_samp = 999;
        for ( int i = 0; i < fbcount; i++ ) {
            XVisualInfo *vi = glXGetVisualFromFBConfig( context.MainDisplay, fbc[i] );
            if (!vi) {
                continue;
            }
            int samp_buf, samples;
            glXGetFBConfigAttrib( context.MainDisplay, fbc[i], GLX_SAMPLE_BUFFERS, &samp_buf );
            glXGetFBConfigAttrib( context.MainDisplay, fbc[i], GLX_SAMPLES       , &samples  );
            if ( best_fbc < 0 || (samp_buf && samples > best_num_samp) )
                best_fbc = i, best_num_samp = samples;
            if ( worst_fbc < 0 || !samp_buf || samples < worst_num_samp )
                worst_fbc = i, worst_num_samp = samples;
            XFree( vi );
        }
        fbc[0] = fbc[ best_fbc ];
    }

    XVisualInfo *visinfo = glXGetVisualFromFBConfig(context.MainDisplay, fbc[0]);
    if (!visinfo)
        pezFatal("Error: couldn't create OpenGL window with this pixel format.\n");

    XSetWindowAttributes attr;
    attr.background_pixel = 0;
    attr.border_pixel = 0;
    attr.colormap = XCreateColormap(context.MainDisplay, root, visinfo->visual, AllocNone);
    attr.event_mask = StructureNotifyMask | ExposureMask | KeyPressMask | KeyReleaseMask |
                      PointerMotionMask | ButtonPressMask | ButtonReleaseMask;

    context.MainWindow = XCreateWindow(
        context.MainDisplay,
        root,
        0, 0,
        PezGetConfig().Width, PezGetConfig().Height, 0,
        visinfo->depth,
        InputOutput,
        visinfo->visual,
        CWBackPixel | CWColormap | CWEventMask,
        &attr
    );

    int borderless = 0;
    if (borderless) {
        Atom mwmHintsProperty = XInternAtom(context.MainDisplay, "_MOTIF_WM_HINTS", 0);
        MwmHints hints = {0};
        hints.flags = MWM_HINTS_DECORATIONS;
        hints.decorations = 0;
        XChangeProperty(context.MainDisplay, context.MainWindow, mwmHintsProperty, mwmHintsProperty, 32,
                        PropModeReplace, (unsigned char *)&hints, PROP_MWM_HINTS_ELEMENTS);
    }

    XMapWindow(context.MainDisplay, context.MainWindow);

    int centerWindow = 1;
    if (centerWindow) {
        Screen* pScreen = XScreenOfDisplay(context.MainDisplay, screenIndex);
        int left = XWidthOfScreen(pScreen)/2 - PezGetConfig().Width/2;
        int top = XHeightOfScreen(pScreen)/2 - PezGetConfig().Height/2;
        XMoveWindow(context.MainDisplay, context.MainWindow, left, top);
    }

    GLXContext glcontext = 0;
    if (PEZ_FORWARD_COMPATIBLE_GL) {
        PFNGLXCREATECONTEXTATTRIBSARBPROC glXCreateContextAttribs = (PFNGLXCREATECONTEXTATTRIBSARBPROC)glXGetProcAddress((GLubyte*)"glXCreateContextAttribsARB");
        if (!glXCreateContextAttribs) {
            pezFatal("Your platform does not support OpenGL 4.0.\n"
                     "Try changing PEZ_FORWARD_COMPATIBLE_GL to 0.\n");
        }
        int attribs[] = {
            GLX_CONTEXT_MAJOR_VERSION_ARB, 4,
            GLX_CONTEXT_MINOR_VERSION_ARB, 0,
            GLX_CONTEXT_FLAGS_ARB, GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
            0
        };
        glcontext = glXCreateContextAttribs(context.MainDisplay, fbc[0], NULL, True, attribs);
    } else {
        glcontext = glXCreateContext(context.MainDisplay, visinfo, NULL, True);
    }

    glXMakeCurrent(context.MainDisplay, context.MainWindow, glcontext);

    glewExperimental = GL_TRUE;
    if(glewInit() != GLEW_OK)
      printf("glewInit didn't work");

    glGetError();

    PFNGLXSWAPINTERVALSGIPROC glXSwapIntervalSGI = (PFNGLXSWAPINTERVALSGIPROC) glXGetProcAddress((GLubyte*)"glXSwapIntervalSGI");
    if (glXSwapIntervalSGI) {
        glXSwapIntervalSGI(PezGetConfig().VerticalSync ? 1 : 0);
    }

    GLenum err = glGetError();

    if(err != GL_NO_ERROR)
      pezCheck(0,gluErrorString(err));

    // Reset OpenGL error state:
    glGetError();

    err = glGetError();

    if(err != GL_NO_ERROR)
      pezCheck(0,gluErrorString(err));

    glGetError();

    err = glGetError();

    if(err != GL_NO_ERROR)
      pezCheck(0,gluErrorString(err));

    // Lop off the trailing .c
    bstring name = bfromcstr(PezGetConfig().Title);
    pezSwInit("");

    // Set up the Shader Wrangler
    pezSwAddPath("./", ".glsl");
    pezSwAddPath("../", ".glsl");
    char qualifiedPath[128];
    strcpy(qualifiedPath, pezResourcePath());
    strcat(qualifiedPath, "/");
    pezSwAddPath(qualifiedPath, ".glsl");
    pezSwAddDirective("*", "#version 400");

    // Perform user-specified intialization
    pezPrintString("OpenGL Version: %s\n", glGetString(GL_VERSION));
    PezInitialize();
    bstring windowTitle = bmidstr(name, 0, blength(name) - 2);
    XStoreName(context.MainDisplay, context.MainWindow, bdata(windowTitle));
    bdestroy(windowTitle);
    bdestroy(name);

    //get keycode of left control, so we can move the inflow with the mouse when it is pressed
    int controlKey = XKeysymToKeycode(context.MainDisplay, XK_Control_L);
    char keys[32];
    bool controlDown = false;

    // -------------------
    // Start the Game Loop
    // -------------------

    unsigned int previousTime = GetMicroseconds();
    int done = 0;
    while (!done) {

        if (glGetError() != GL_NO_ERROR)
            pezFatal("OpenGL error.\n");

        while (XPending(context.MainDisplay)) {
            XEvent event;

            XNextEvent(context.MainDisplay, &event);
            switch (event.type)
            {
                case Expose:
                    break;

                case ConfigureNotify:
                    break;

#ifdef PEZ_MOUSE_HANDLER
                case ButtonPress:

                    XQueryKeymap(context.MainDisplay, keys);
                    if(keys[controlKey/8]&(0x1<<(controlKey%8)))
                        controlDown = true;
                    else
                        controlDown = false;
                    PezHandleMouse(event.xbutton.x, event.xbutton.y, PEZ_DOWN, controlDown);
                    break;

                case ButtonRelease:

                    XQueryKeymap(context.MainDisplay, keys);

                    if(keys[controlKey/8]&(0x1<<(controlKey%8)))
                        controlDown = true;
                    else
                        controlDown = false;
                    PezHandleMouse(event.xbutton.x, event.xbutton.y, PEZ_UP, controlDown);
                    break;

                case MotionNotify:
                    XQueryKeymap(context.MainDisplay, keys);
                    if(keys[controlKey/8]&(0x1<<(controlKey%8))){
                        controlDown = true;
                    }
                    else
                        controlDown = false;
                    PezHandleMouse(event.xmotion.x, event.xmotion.y, PEZ_MOVE, controlDown);
                    break;
#endif

                case KeyRelease: {
                    XComposeStatus composeStatus;
                    char asciiCode[32];
                    KeySym keySym;
                    int len;

                    len = XLookupString(&event.xkey, asciiCode, sizeof(asciiCode), &keySym, &composeStatus);
                    switch (asciiCode[0]) {
                        case 'x': case 'X': case 'q': case 'Q':
                        case 0x1b:
                            done = 1;
                            break;
                        default:
                            PezHandleKey(asciiCode[0]);
                    }
                }
            }
        }

        unsigned int currentTime = GetMicroseconds();
        unsigned int deltaTime = currentTime - previousTime;
        previousTime = currentTime;

        PezUpdate((float) deltaTime / 1000000.0f);

        PezRender();
        glXSwapBuffers(context.MainDisplay, context.MainWindow);
    }

    pezSwShutdown();

    return 0;
}

void pezPrintStringW(const wchar_t* pStr, ...)
{
    va_list a;
    va_start(a, pStr);

    wchar_t msg[1024] = {0};
    vswprintf(msg, countof(msg), pStr, a);
    fputws(msg, stderr);
}

void pezPrintString(const char* pStr, ...)
{
    va_list a;
    va_start(a, pStr);

    char msg[1024] = {0};
    vsnprintf(msg, countof(msg), pStr, a);
    fputs(msg, stderr);
}

void pezFatalW(const wchar_t* pStr, ...)
{
    fwide(stderr, 1);

    va_list a;
    va_start(a, pStr);

    wchar_t msg[1024] = {0};
    vswprintf(msg, countof(msg), pStr, a);
    fputws(msg, stderr);
    exit(1);
}

void _pezFatal(const char* pStr, va_list a)
{
    char msg[1024] = {0};
    vsnprintf(msg, countof(msg), pStr, a);
    fputs(msg, stderr);
    fputc('\n', stderr);
    exit(1);
}

void pezFatal(const char* pStr, ...)
{
    va_list a;
    va_start(a, pStr);
    _pezFatal(pStr, a);
}

void pezCheck(int condition, ...)
{
    va_list a;
    const char* pStr;

    if (condition)
        return;

    va_start(a, condition);
    pStr = va_arg(a, const char*);
    _pezFatal(pStr, a);
}

void pezCheckPointer(void* p, ...)
{
    va_list a;
    const char* pStr;

    if (p != NULL)
        return;

    va_start(a, p);
    pStr = va_arg(a, const char*);
    _pezFatal(pStr, a);
}

int pezIsPressing(char key)
{
    return 0;
}

const char* pezResourcePath()
{
    return ".";
}
