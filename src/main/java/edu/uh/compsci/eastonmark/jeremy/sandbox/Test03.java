package edu.uh.compsci.eastonmark.jeremy.sandbox;

import org.opencv.core.Core;

public class Test03 {
	static {
        nu.pattern.OpenCV.loadShared();
        // nu.pattern.OpenCV.loadLocally(); // Use in case loadShared() doesn't work
    }

    public static void main(String[] args) {
        System.out.println(Core.VERSION);
        System.out.println(Core.VERSION_MAJOR);
        System.out.println(Core.VERSION_MINOR);
        System.out.println(Core.VERSION_REVISION);
        System.out.println(Core.NATIVE_LIBRARY_NAME);
        System.out.println(Core.getBuildInformation());
    }
}
